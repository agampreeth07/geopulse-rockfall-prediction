# ⛰️ GeoPulse

## Rockfall Prediction & Risk Assessment

GeoPulse is a research-oriented machine learning prototype designed to estimate rockfall risk from environmental and geological conditions.

The system uses a **Random Forest classifier** to analyze factors such as slope angle, soil moisture, rainfall, previous rockfall history, and temperature.

## 🚀 Live Demo

**Try GeoPulse online:**

https://geopulse-rockfall-prediction.streamlit.app/

---

## 📌 Project Overview

Rockfalls are influenced by multiple environmental and geological conditions. Early identification of potentially hazardous conditions can support monitoring and risk-assessment workflows.

GeoPulse explores how machine learning can be applied to environmental indicators to estimate the likelihood of rockfall events.

The project combines:

- Data preprocessing
- Machine learning
- Random Forest classification
- Model evaluation
- Risk prediction
- Data visualization
- Streamlit deployment

---

## 🎯 Objectives

The main objectives of GeoPulse are:

1. Analyze environmental and geological factors associated with rockfall events.
2. Develop a machine learning classification model.
3. Estimate rockfall risk from user-provided conditions.
4. Evaluate the model using classification metrics.
5. Visualize model performance and feature importance.
6. Provide an interactive web interface using Streamlit.

---

## 🧠 Machine Learning Model

GeoPulse currently uses a **Random Forest Classifier**.

Random Forest was selected because it can model nonlinear relationships between environmental variables and the target classification while also providing feature-importance information.

### Input Features

| Feature | Description |
|---|---|
| Slope Angle | Slope inclination in degrees |
| Soil Moisture | Soil moisture percentage |
| Rainfall | Rainfall amount in millimeters |
| Previous Rockfall History | Indicates whether previous rockfall events occurred |
| Temperature | Temperature in degrees Celsius |

### Target

The model predicts whether the given environmental conditions indicate:

- **ROCKFALL RISK**
- **NO ROCKFALL RISK**

---

## 🔄 System Workflow

```text
Environmental Conditions
          ↓
      Data Loading
          ↓
       Preprocessing
          ↓
   Train/Test Split
          ↓
   Random Forest Model
          ↓
       Prediction
          ↓
   Risk Probability
          ↓
   Risk Classification
          ↓
   Streamlit Dashboard
