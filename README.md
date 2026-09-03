# ⛰️ GeoPulse

## Rockfall Prediction & Risk Assessment

GeoPulse is a research-oriented machine learning prototype designed to estimate rockfall risk from environmental and geological conditions.

The system uses a **Random Forest Classifier** to analyze factors such as slope angle, soil moisture, rainfall, previous rockfall history, and temperature.

---

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

## 🧠 Machine Learning Model

GeoPulse currently uses a **Random Forest Classifier**.

Random Forest was selected because it can model nonlinear relationships between environmental variables and the target classification while also providing feature-importance information.

### Input Features

| Feature | Description |
| --- | --- |
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

   ---

# 📊 Model Performance

GeoPulse was evaluated using a synthetic dataset containing **1,500 samples**.

| Metric | Result |
| --- | --- |
| Dataset Size | 1,500 samples |
| Training Samples | 1,200 |
| Testing Samples | 300 |
| Rockfall Risk Samples | 900 |
| Safe Samples | 600 |
| Accuracy | 69.3% |
| ROC-AUC | 0.778 |

### Classification Report

| Class | Precision | Recall | F1-Score | Support |
| --- | ---: | ---: | ---: | ---: |
| Safe | 0.61 | 0.63 | 0.62 | 120 |
| Rockfall Risk | 0.75 | 0.73 | 0.74 | 180 |
| Overall Accuracy | | | **0.69** | **300** |

### Confusion Matrix

| Actual / Predicted | Safe | Rockfall Risk |
| --- | ---: | ---: |
| Safe | 76 | 44 |
| Rockfall Risk | 48 | 132 |

The results indicate that the Random Forest model can distinguish between safe and rockfall-risk conditions within the synthetic evaluation dataset.

The ROC-AUC score of **0.778** indicates useful classification ability above random guessing.

> **Important:** These results are based on a synthetic dataset created for research and demonstration purposes. They should not be interpreted as real-world rockfall prediction accuracy or as a safety-critical warning system.

---

## 📈 Visualizations

The project generates the following evaluation visualizations:

- Confusion Matrix
- ROC Curve
- Feature Importance

These files are available in the `results/` directory:

```text
results/
├── confusion_matrix.png
├── roc_curve.png
└── feature_importance.png

---

## 📁 Dataset

GeoPulse currently uses a **synthetic dataset** created for research and prototype evaluation.

The dataset contains:

- **1,500 total samples**
- **900 rockfall-risk samples**
- **600 non-rockfall samples**
- 5 environmental/geological input features
- 1 binary target variable

Dataset file:

```text
data/raw/geopulse_synthetic_dataset.csv

geopulse-rockfall-prediction/
│
├── app.py
├── requirements.txt
├── README.md
│
├── data/
│   └── raw/
│       ├── geopulse_synthetic_dataset.csv
│       └── rockfall_data.csv
│
├── models/
│   └── geopulse_random_forest.pkl
│
├── results/
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── feature_importance.png
│
├── src/
│   ├── data_loader.py
│   ├── preprocess.py
│   ├── train.py
│   ├── predict.py
│   ├── evaluate.py
│   └── visualize.py
│
└── research/

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/agampreeth07/geopulse-rockfall-prediction.git
cd geopulse-rockfall-prediction
