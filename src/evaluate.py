from pathlib import Path

import joblib
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from preprocess import prepare_data


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "geopulse_random_forest.pkl"


def evaluate_model():
    """Evaluate the trained GeoPulse Random Forest model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}. "
            "Run train.py first."
        )

    model = joblib.load(MODEL_PATH)

    X_train, X_test, y_train, y_test = prepare_data()

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    roc_auc = roc_auc_score(y_test, probabilities)

    print("\n" + "=" * 50)
    print("GeoPulse Model Evaluation")
    print("=" * 50)

    print(f"\nAccuracy : {accuracy:.3f}")
    print(f"ROC-AUC  : {roc_auc:.3f}")

    print("\nClassification Report:")
    print(classification_report(
        y_test,
        predictions,
        target_names=["Safe", "Rockfall Risk"],
        zero_division=0,
    ))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))


if __name__ == "__main__":
    evaluate_model()