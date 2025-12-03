import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from model import train

st.set_page_config(page_title="Bitcoin Predictor Pro", layout="wide")

st.title("Bitcoin Price Prediction PRO")
st.write("Advanced ML forecasting with visualization dashboard.")

uploaded = st.file_uploader("Upload cleaned Bitcoin CSV", type=["csv"])

model_choice = st.selectbox(
    "Choose ML model",
    ["Linear Regression", "Random Forest", "XGBoost"]
)

forecast_days = st.slider("How many days to predict?", 3, 30, 7)

df = None
valid_csv = False

if uploaded:
    df = pd.read_csv(uploaded)
    df.to_csv("cleaned_bitcoin.csv", index=False)

    st.success("CSV loaded successfully.")

    # Validate required columns
    required_columns = ["Date", "Price"]
    valid_csv = all(col in df.columns for col in required_columns)
    if not valid_csv:
        st.error(
            f"CSV must contain columns: {', '.join(required_columns)}. "
            "Optional: Open, High, Low for candlestick."
        )
    else:
        st.subheader("Price chart")
        if all(col in df.columns for col in ["Open", "High", "Low"]):
            fig = go.Figure(
                data=[
                    go.Candlestick(
                        x=df["Date"],
                        open=df["Open"],
                        high=df["High"],
                        low=df["Low"],
                        close=df["Price"],
                        name="BTC"
                    )
                ]
            )
        else:
            fig = go.Figure(
                data=[
                    go.Scatter(
                        x=df["Date"],
                        y=df["Price"],
                        mode="lines",
                        name="BTC"
                    )
                ]
            )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

if st.button("Run Forecast"):
    if df is None:
        st.error("Please upload a cleaned CSV first.")
    elif not valid_csv:
        st.error("Uploaded CSV is missing required columns. Fix the CSV and upload again.")
    else:
        try:
            score, forecast = train(df, model_choice, forecast_days)
        except Exception as e:
            st.error(f"Error during training: {e}")
        else:
            st.subheader("Model accuracy (R²)")
            st.metric("R² score", f"{round(score, 4)}")

            st.subheader(f"Next {forecast_days} day predictions")
            st.write(forecast)
