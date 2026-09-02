from pathlib import Path
import pandas as pd


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Dataset location
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "geopulse_synthetic_dataset.csv"


def load_data():
    """Load the GeoPulse synthetic dataset from the raw data folder."""

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {DATA_PATH}"
        )

    data = pd.read_csv(DATA_PATH)

    print("Dataset loaded successfully.")
    print(f"Dataset shape: {data.shape}")
    print("\nColumns:")
    print(list(data.columns))

    return data


if __name__ == "__main__":
    df = load_data()

    print("\nFirst 5 rows:")
    print(df.head())