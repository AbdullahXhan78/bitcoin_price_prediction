import pandas as pd

def clean_csv(input_file, output_file="cleaned_bitcoin.csv"):
    df = pd.read_csv(input_file)

    # Clean numeric columns if they exist
    numeric_cols = ["Price", "Open", "High", "Low", "Vol.", "Change %"]
    for col in numeric_cols:
        if col in df.columns:
            if col in ["Vol."]:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.replace("K", "000")
                    .str.replace("M", "000000")
                    .str.replace("B", "000000000")
                    .str.replace(",", "")
                )
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
            elif col == "Change %":
                df[col] = pd.to_numeric(df[col].astype(str).str.replace("%", ""), errors="coerce")
            else:
                df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values("Date").dropna(subset=["Date"])  # Ensure valid dates

    df.to_csv(output_file, index=False)
    print("Cleaned file saved:", output_file)

if __name__ == "__main__":
    # Example: clean_csv("your_input.csv")
    pass  # Call with your file, e.g., from command line
