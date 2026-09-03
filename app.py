```python
import streamlit as st
import sys
from pathlib import Path
from PIL import Image

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_PATH = PROJECT_ROOT / "src"
RESULTS_PATH = PROJECT_ROOT / "results"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from predict import predict_risk


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="GeoPulse | Rockfall Prediction",
    page_icon="⛰️",
    layout="wide",
)


# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.title("⛰️ GeoPulse")

st.subheader("Rockfall Prediction & Risk Assessment")

st.write(
    "GeoPulse estimates rockfall risk using environmental "
    "and geological conditions with a trained Random Forest model."
)

st.divider()


# --------------------------------------------------
# ENVIRONMENTAL INPUTS
# --------------------------------------------------

st.header("🌍 Environmental Conditions")

col1, col2 = st.columns(2)


# --------------------------------------------------
# LEFT COLUMN
# --------------------------------------------------

with col1:

    slope_angle = st.number_input(
        "Slope Angle (degrees)",
        min_value=0.0,
        max_value=90.0,
        value=45.0,
        step=1.0,
    )

    soil_moisture = st.number_input(
        "Soil Moisture (%)",
        min_value=0.0,
        max_value=100.0,
        value=55.0,
        step=1.0,
    )

    rainfall = st.number_input(
        "Rainfall (mm)",
        min_value=0.0,
        max_value=500.0,
        value=60.0,
        step=1.0,
    )


# --------------------------------------------------
# RIGHT COLUMN
# --------------------------------------------------

with col2:

    rockfall_history = st.selectbox(
        "Previous Rockfall History",
        options=[0, 1],
        format_func=lambda x:
            "No previous rockfall"
            if x == 0
            else "Previous rockfall recorded",
    )

    temperature = st.number_input(
        "Temperature (°C)",
        min_value=-50.0,
        max_value=60.0,
        value=21.0,
        step=1.0,
    )

st.divider()


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if st.button(
    "🔍 Predict Rockfall Risk",
    type="primary",
    use_container_width=True,
):

    probability, risk_level, prediction = predict_risk(
        slope_angle=slope_angle,
        soil_moisture=soil_moisture,
        rainfall=rainfall,
        rockfall_history=rockfall_history,
        temperature=temperature,
    )

    st.header("📊 Prediction Result")

    result_col1, result_col2, result_col3 = st.columns(3)


    # --------------------------------------------------
    # RISK PROBABILITY
    # --------------------------------------------------

    with result_col1:

        st.metric(
            "Risk Probability",
            f"{probability:.2%}",
        )


    # --------------------------------------------------
    # RISK LEVEL
    # --------------------------------------------------

    with result_col2:

        st.metric(
            "Risk Level",
            risk_level,
        )


    # --------------------------------------------------
    # PREDICTION
    # --------------------------------------------------

    with result_col3:

        st.metric(
            "Prediction",
            prediction,
        )


    # --------------------------------------------------
    # PROBABILITY BAR
    # --------------------------------------------------

    st.progress(
        min(int(probability * 100), 100)
    )


    # --------------------------------------------------
    # RISK MESSAGE
    # --------------------------------------------------

    if risk_level == "HIGH":

        st.error(
            "⚠️ HIGH RISK: The estimated rockfall probability "
            "is above the configured threshold."
        )

    elif risk_level == "MEDIUM":

        st.warning(
            "⚠️ MEDIUM RISK: Potential rockfall risk detected. "
            "Continue monitoring conditions."
        )

    else:

        st.success(
            "✓ LOW RISK: Current conditions indicate lower "
            "estimated rockfall risk."
        )


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

st.divider()

st.header("📈 Model Performance")

st.info(
    "Prototype evaluation using a synthetic dataset of 1,500 samples, "
    "with 300 held-out test samples. These results are for research "
    "and development only and should not be interpreted as real-world "
    "safety accuracy."
)


# --------------------------------------------------
# PERFORMANCE METRICS
# --------------------------------------------------

metric1, metric2 = st.columns(2)


with metric1:

    st.metric(
        "Accuracy",
        "69.3%",
    )


with metric2:

    st.metric(
        "ROC-AUC",
        "0.778",
    )


# --------------------------------------------------
# VISUALIZATION RESULTS
# --------------------------------------------------

st.subheader("Model Evaluation Visualizations")

image_col1, image_col2 = st.columns(2)


# --------------------------------------------------
# RESULT FILE PATHS
# --------------------------------------------------

confusion_matrix_path = (
    RESULTS_PATH / "confusion_matrix.png"
)

roc_curve_path = (
    RESULTS_PATH / "roc_curve.png"
)

feature_importance_path = (
    RESULTS_PATH / "feature_importance.png"
)


# --------------------------------------------------
# LEFT VISUALIZATION COLUMN
# --------------------------------------------------

with image_col1:

    if confusion_matrix_path.exists():

        st.image(
            Image.open(confusion_matrix_path),
            caption="Confusion Matrix",
            use_container_width=True,
        )

    else:

        st.warning(
            "Confusion matrix image not found."
        )

    if roc_curve_path.exists():

        st.image(
            Image.open(roc_curve_path),
            caption="ROC Curve",
            use_container_width=True,
        )

    else:

        st.warning(
            "ROC curve image not found."
        )


# --------------------------------------------------
# RIGHT VISUALIZATION COLUMN
# --------------------------------------------------

with image_col2:

    if feature_importance_path.exists():

        st.image(
            Image.open(feature_importance_path),
            caption="Feature Importance",
            use_container_width=True,
        )

    else:

        st.warning(
            "Feature importance image not found."
        )


# --------------------------------------------------
# ABOUT GEOPULSE
# --------------------------------------------------

st.divider()

st.header("ℹ️ About GeoPulse")

st.write(
    "GeoPulse is a research-oriented rockfall prediction "
    "prototype that uses environmental and geological "
    "indicators to estimate rockfall risk."
)

st.write(
    "**Current input factors:** slope angle, soil moisture, "
    "rainfall, previous rockfall history, and temperature."
)


# --------------------------------------------------
# DATASET INFORMATION
# --------------------------------------------------

st.subheader("🧪 Dataset")

st.write(
    "The current prototype was evaluated using a synthetic "
    "dataset containing 1,500 samples and five environmental "
    "and geological input features."
)

st.write(
    "The dataset contains 900 rockfall-risk samples and "
    "600 non-rockfall samples."
)


# --------------------------------------------------
# RESEARCH NOTICE
# --------------------------------------------------

st.caption(
    "Research prototype: The dataset is synthetic and the "
    "reported model performance is intended for development "
    "and research evaluation only."
)


# --------------------------------------------------
# SAFETY DISCLAIMER
# --------------------------------------------------

st.warning(
    "⚠️ Safety Disclaimer: GeoPulse is not a certified "
    "rockfall warning or emergency decision system. "
    "Predictions should not be used as a substitute for "
    "professional geological assessment, field monitoring, "
    "or emergency-management procedures."
)
```
