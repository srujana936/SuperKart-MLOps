
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"

TARGET = "Product_Store_Sales_Total"

EXPECTED_COLUMNS = [
    "Product_Id",
    "Product_Weight",
    "Product_Sugar_Content",
    "Product_Allocated_Area",
    "Product_Type",
    "Product_MRP",
    "Store_Id",
    "Store_Establishment_Year",
    "Store_Size",
    "Store_Location_City_Type",
    "Store_Type",
    TARGET
]

def main():
    source_path = DATA_DIR / "superkart.csv"

    if not source_path.exists():
        raise FileNotFoundError(f"Dataset not found: {source_path}")

    df = pd.read_csv(source_path)
    df.columns = df.columns.str.strip()

    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Retain only the documented project columns.
    df = df[EXPECTED_COLUMNS].copy()

    # Clean whitespace in categorical/text fields.
    text_columns = df.select_dtypes(include=["object", "string"]).columns
    for col in text_columns:
        df[col] = df[col].map(
            lambda value: value.strip() if isinstance(value, str) else value
        )
        df[col] = df[col].replace("", pd.NA)

    # Remove exact duplicate records.
    before = len(df)
    df = df.drop_duplicates()
    print("Duplicate rows removed:", before - len(df))

    # The target must be numeric and present.
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    before = len(df)
    df = df.dropna(subset=[TARGET])
    print("Rows removed for missing/invalid target:", before - len(df))

    if df.empty:
        raise ValueError("No usable records remain after cleaning.")

    # Target is sales, so negative target values should be investigated.
    negative_targets = int((df[TARGET] < 0).sum())
    print("Negative target values requiring investigation:", negative_targets)

    # Do not silently remove negative values; verify them against the data
    # dictionary or business rules before deciding how to handle them.

    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(DATA_DIR / "superkart_train.csv", index=False)
    test_df.to_csv(DATA_DIR / "superkart_test.csv", index=False)

    print("Clean dataset shape:", df.shape)
    print("Training dataset shape:", train_df.shape)
    print("Testing dataset shape:", test_df.shape)
    print("Saved training data:", DATA_DIR / "superkart_train.csv")
    print("Saved testing data:", DATA_DIR / "superkart_test.csv")

if __name__ == "__main__":
    main()
