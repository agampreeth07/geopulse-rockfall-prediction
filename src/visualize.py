from pathlib import Path

import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay, confusion_matrix
from sklearn.model_selection import train_test_split

from data_loader import load_data


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "geopulse_random_forest.pkl"
RESULTS_DIR = PROJECT_ROOT / "results"

FEATURE_COLUMNS = [
    "slope_angle",
    "soil_moisture",
    "rainfall",
    "rockfall_history",
    "temperature",
]

TARGET_COLUMN = "rockfall"


def prepare_test_data():
    """Prepare the same test split used during model training."""

    data = load_data()

    X = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    _, X_test, _, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    return X_test, y_test


def create_visualizations():
    """Generate model evaluation visualizations."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}. Run train.py first."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    model = joblib.load(MODEL_PATH)
    X_test, y_test = prepare_test_data()

    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    # 1. Confusion Matrix
    cm = confusion_matrix(y_test, predictions)

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Safe", "Rockfall Risk"],
    )

    disp.plot()
    plt.title("GeoPulse Confusion Matrix")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=300)
    plt.close()

    # 2. ROC Curve
    RocCurveDisplay.from_predictions(
        y_test,
        probabilities,
    )

    plt.title("GeoPulse ROC Curve")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "roc_curve.png", dpi=300)
    plt.close()

    # 3. Feature Importance
    importance = model.feature_importances_

    plt.figure(figsize=(8, 5))
    plt.barh(FEATURE_COLUMNS, importance)
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.title("GeoPulse Random Forest Feature Importance")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "feature_importance.png", dpi=300)
    plt.close()

    print("\nVisualizations generated successfully.")

    print("\nFiles created:")
    print(f"- {RESULTS_DIR / 'confusion_matrix.png'}")
    print(f"- {RESULTS_DIR / 'roc_curve.png'}")
    print(f"- {RESULTS_DIR / 'feature_importance.png'}")


if __name__ == "__main__":
    create_visualizations()