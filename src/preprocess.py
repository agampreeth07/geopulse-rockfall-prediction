from sklearn.model_selection import train_test_split

from data_loader import load_data


FEATURE_COLUMNS = [
    "slope_angle",
    "soil_moisture",
    "rainfall",
    "rockfall_history",
    "temperature",
]

TARGET_COLUMN = "rockfall"


def prepare_data():
    """Load the dataset and split it into training and testing data."""

    data = load_data()

    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    print("\nPreprocessing completed.")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    prepare_data()