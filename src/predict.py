from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "geopulse_random_forest.pkl"

FEATURE_COLUMNS = [
    "slope_angle",
    "soil_moisture",
    "rainfall",
    "rockfall_history",
    "temperature",
]

RISK_THRESHOLD = 0.70


def load_model():
    """Load the trained GeoPulse model."""

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found at: {MODEL_PATH}. "
            "Run train.py first."
        )

    return joblib.load(MODEL_PATH)


def predict_risk(
    slope_angle,
    soil_moisture,
    rainfall,
    rockfall_history,
    temperature,
):
    """Predict rockfall risk for a single observation."""

    model = load_model()

    input_data = pd.DataFrame(
        [[
            slope_angle,
            soil_moisture,
            rainfall,
            rockfall_history,
            temperature,
        ]],
        columns=FEATURE_COLUMNS,
    )

    probability = model.predict_proba(input_data)[0][1]

    if probability >= RISK_THRESHOLD:
        risk_level = "HIGH"
        prediction = "ROCKFALL RISK"
    elif probability >= 0.40:
        risk_level = "MEDIUM"
        prediction = "POTENTIAL RISK"
    else:
        risk_level = "LOW"
        prediction = "SAFE"

    return probability, risk_level, prediction


if __name__ == "__main__":

    # Example test observation
    probability, risk_level, prediction = predict_risk(
        slope_angle=45,
        soil_moisture=55,
        rainfall=60,
        rockfall_history=1,
        temperature=21,
    )

    print("\n" + "=" * 50)
    print("GeoPulse Rockfall Prediction")
    print("=" * 50)

    print(f"\nRisk Probability : {probability:.2%}")
    print(f"Risk Level       : {risk_level}")
    print(f"Prediction       : {prediction}")

    if probability >= RISK_THRESHOLD:
        print("\nALERT: Rockfall risk is above the threshold.")
    else:
        print("\nMonitoring continues. Risk is below the threshold.")