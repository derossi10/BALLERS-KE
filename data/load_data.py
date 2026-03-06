import pandas as pd
from pathlib import Path

# Paths
RAW_DATA_PATH = Path("data/raw/kl.csv")
PROCESSED_DATA_PATH = Path("data/processed/players_clean.csv")

def load_raw_data():
    """Load raw dataset with proper encoding."""
    df = pd.read_csv(RAW_DATA_PATH, encoding="latin1")
    return df

def clean_money(value):
    """Convert money values like €110.5M or €200K to numeric."""
    
    if isinstance(value, str):
        value = value.replace("€", "")
        value = value.replace("\x80", "")   # remove strange euro encoding
        value = value.strip()

        if "M" in value:
            return float(value.replace("M", "")) * 1_000_000
        elif "K" in value:
            return float(value.replace("K", "")) * 1_000
        else:
            try:
                return float(value)
            except:
                return None
    return value

def clean_dataset(df):

    df["Value"] = df["Value"].apply(clean_money)
    df["Wage"] = df["Wage"].apply(clean_money)

    # Position grouping
    def position_group(pos):
        if pos in ["GK"]:
            return "GK"
        if pos in ["CB","LB","RB","LWB","RWB"]:
            return "DEF"
        if pos in ["CDM","CM","CAM","LM","RM"]:
            return "MID"
        return "FWD"

    df["PositionGroup"] = df["Position"].apply(position_group)

    return df

def save_clean_data(df):

    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_PATH, index=False)

def main():

    print("Loading dataset...")
    df = load_raw_data()

    print("Cleaning dataset...")
    df = clean_dataset(df)

    print("Saving cleaned dataset...")
    save_clean_data(df)

    print("Done. Clean dataset saved.")

if __name__ == "__main__":
    main()