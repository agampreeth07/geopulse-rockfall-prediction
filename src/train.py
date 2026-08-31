from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier

from preprocess import prepare_data


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "geopulse_random_forest.pkl"


def train_model():
    """Train and save the GeoPulse Random Forest model."""

    X_train, X_test, y_train, y_test = prepare_data()

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("\nRandom Forest training completed.")
    print(f"Model saved to: {MODEL_PATH}")

    return model, X_test, y_test


if __name__ == "__main__":
    train_model()