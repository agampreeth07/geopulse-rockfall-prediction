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
    """Load the trained GeoPulse Random Forest model."""

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
    """Predict rockfall risk for one observation."""

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


def get_number(prompt):
    """Safely get a numeric value from the user."""

    while True:
        try:
            return float(input(prompt))
        except ValueError:
            print("Please enter a valid number.")


def main():
    print("\n" + "=" * 50)
    print("        GeoPulse Rockfall Prediction")
    print("=" * 50)

    print("\nEnter the environmental conditions:\n")

    slope_angle = get_number(
        "Slope angle (degrees): "
    )

    soil_moisture = get_number(
        "Soil moisture (%): "
    )

    rainfall = get_number(
        "Rainfall (mm): "
    )

    rockfall_history = get_number(
        "Rockfall history (0 = No, 1 = Yes): "
    )

    temperature = get_number(
        "Temperature (°C): "
    )

    probability, risk_level, prediction = predict_risk(
        slope_angle=slope_angle,
        soil_moisture=soil_moisture,
        rainfall=rainfall,
        rockfall_history=rockfall_history,
        temperature=temperature,
    )

    print("\n" + "=" * 50)
    print("             PREDICTION RESULT")
    print("=" * 50)

    print(f"\nRisk Probability : {probability:.2%}")
    print(f"Risk Level       : {risk_level}")
    print(f"Prediction       : {prediction}")

    if risk_level == "HIGH":
        print("\n⚠️ ALERT: Rockfall risk is above the threshold.")
    elif risk_level == "MEDIUM":
        print("\n⚠️ WARNING: Potential rockfall risk detected.")
    else:
        print("\n✓ Conditions currently indicate lower risk.")

    print("\n" + "=" * 50)


if __name__ == "__main__":
    main()