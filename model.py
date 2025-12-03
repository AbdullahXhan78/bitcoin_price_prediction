import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

try:
    from xgboost import XGBRegressor
except Exception:  # xgboost may be missing; handled in code
    XGBRegressor = None


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure numeric columns are numeric and dates are parsed."""
    df = df.copy()
    # Standard expected columns
    numeric_cols = ["Price", "Open", "High", "Low"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Date" not in df.columns:
        raise ValueError("CSV must include a 'Date' column.")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    
    # Validate that we have valid datetime objects
    if df["Date"].isna().all():
        raise ValueError("No valid dates found in the Date column after conversion.")
    
    df = df.dropna(subset=["Date", "Price"]).sort_values("Date")
    df = df.ffill().bfill()  # smooth small gaps
    return df


def _build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Create model features from the cleaned dataframe."""
    features = pd.DataFrame(index=df.index)
    
    # Validate that Date column contains datetime objects
    if not pd.api.types.is_datetime64_any_dtype(df["Date"]):
        raise ValueError("Date column must contain datetime objects. Call _normalize_columns first.")
    
    # Use available OHLC columns; fallback to price only
    for col in ["Price", "Open", "High", "Low"]:
        if col in df.columns:
            features[col] = df[col]
    
    # Extract datetime features using explicit function calls to avoid Pylance issues
    dates = df["Date"]
    features["day_of_year"] = dates.apply(lambda x: x.timetuple().tm_yday if pd.notna(x) else 0)
    features["month"] = dates.apply(lambda x: x.month if pd.notna(x) else 1)
    features["year"] = dates.apply(lambda x: x.year if pd.notna(x) else 2000)
    features["dayofweek"] = dates.apply(lambda x: x.weekday() if pd.notna(x) else 0)
    return features


def prepare(df: pd.DataFrame, forecast_out: int):
    """
    Prepare training and forecasting sets.

    Returns:
        X_train, X_test, y_train, y_test, future_features, future_dates
    """
    if forecast_out < 1:
        raise ValueError("forecast_out must be >= 1")

    df = _normalize_columns(df)
    features = _build_feature_frame(df)

    # Shift label to create future target
    labels = df["Price"].shift(-forecast_out)
    train_mask = labels.notna()

    if train_mask.sum() <= forecast_out:
        raise ValueError(f"Not enough rows for forecasting: need more than {forecast_out} rows.")

    X = features[train_mask]
    y = labels[train_mask]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    # Build naive future features using the last known OHLC values
    last_row = df.iloc[-1]
    future_dates = pd.date_range(start=last_row["Date"] + pd.Timedelta(days=1), periods=forecast_out, freq="D")
    future_df = pd.DataFrame({
        "Date": future_dates,
        "Price": last_row["Price"]
    })
    for col in ["Open", "High", "Low"]:
        if col in df.columns:
            future_df[col] = last_row[col]
    future_features = _build_feature_frame(future_df)

    return X_train, X_test, y_train, y_test, future_features, future_dates


def _get_model(choice: str):
    choice = choice.lower()
    if choice == "linear regression":
        return make_pipeline(StandardScaler(), LinearRegression())
    if choice == "random forest":
        return RandomForestRegressor(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        )
    if choice == "xgboost":
        if XGBRegressor is None:
            raise ImportError("xgboost is not installed. Install it or choose another model.")
        return XGBRegressor(
            objective="reg:squarederror",
            n_estimators=400,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
    raise ValueError(f"Unknown model choice: {choice}")


def train(df: pd.DataFrame, model_choice: str, forecast_days: int):
    """
    Train the selected model and produce forecasts.

    Returns:
        score (float): R² on held-out data
        forecast_df (pd.DataFrame): future dates with predicted prices
    """
    X_train, X_test, y_train, y_test, future_features, future_dates = prepare(df, forecast_days)

    model = _get_model(model_choice)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    score = r2_score(y_test, preds)

    future_preds = model.predict(future_features)
    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted_Price": future_preds
    })

    return score, forecast_df
