"""Global Air Pollution: AQI Prediction System (Streamlit web app, Assignment 3).

Course : AIC354 Machine Learning Fundamentals, COMSATS University Islamabad, Lahore Campus
Student: Haziq Mahmood (FA24-BSE-049)

The app loads the trained scikit-learn pipeline saved by the notebook
(aqi_prediction_kfold_full_dataset_10fold.ipynb) and predicts the overall Air Quality
Index (AQI) of a city from the AQI values of three pollutants (CO, Ozone, NO2) and its
country, without using the PM2.5 reading.
"""
import csv
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "aqi_prediction_model.pkl"
INFO_PATH = BASE_DIR / "model_info.json"
DATA_PATH = BASE_DIR / "global_air_pollution_dataset.csv"
FEEDBACK_PATH = BASE_DIR / "feedback.csv"

NUMERIC = ["CO AQI Value", "Ozone AQI Value", "NO2 AQI Value"]
CATEGORICAL = ["Country"]
FEATURES = NUMERIC + CATEGORICAL

AQI_BANDS = [  # upper limit, category, message type
    (50, "Good", "success"),
    (100, "Moderate", "info"),
    (150, "Unhealthy for Sensitive Groups", "warning"),
    (200, "Unhealthy", "error"),
    (300, "Very Unhealthy", "error"),
    (500, "Hazardous", "error"),
]


def aqi_category(value):
    for limit, name, kind in AQI_BANDS:
        if value <= limit:
            return name, kind
    return AQI_BANDS[-1][1], AQI_BANDS[-1][2]


def train_fallback_model(algorithm_text):
    """Re-create the notebook's final model from the CSV if the .pkl cannot be loaded
    (for example because the cloud machine has a different scikit-learn version)."""
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
    from sklearn.linear_model import Lasso, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.svm import SVR

    algorithms = {
        "Ridge Regression": Ridge(alpha=1.0),
        "Lasso Regression": Lasso(alpha=0.1),
        "Support Vector (SVR)": SVR(kernel="rbf", C=100, epsilon=1.0),
        "Extra Trees": ExtraTreesRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(random_state=42),
    }
    chosen = next((a for name, a in algorithms.items() if algorithm_text.startswith(name)),
                  algorithms["Extra Trees"])

    data = pd.read_csv(DATA_PATH).dropna()
    pipeline = Pipeline([
        ("preprocessor", ColumnTransformer([
            ("numeric", StandardScaler(), NUMERIC),
            ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL),
        ])),
        ("regressor", chosen),
    ])
    return pipeline.fit(data[FEATURES], data["AQI Value"])


@st.cache_data(show_spinner=False)
def load_info():
    return json.loads(INFO_PATH.read_text(encoding="utf-8"))


@st.cache_resource(show_spinner="Loading the model...")
def load_model():
    """Load the saved model (no re-training); fall back to re-creating it if loading fails."""
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except Exception:
        return train_fallback_model(load_info()["model"])


def save_feedback(rating, comment, inputs, prediction):
    """Append one feedback row to feedback.csv. Returns True on success."""
    try:
        is_new = not FEEDBACK_PATH.exists()
        with open(FEEDBACK_PATH, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if is_new:
                writer.writerow(["timestamp", "rating", "comment", *FEATURES, "predicted_aqi"])
            writer.writerow([datetime.now().isoformat(timespec="seconds"), rating, comment.strip(),
                             *[inputs[k] for k in FEATURES], round(prediction, 1)])
        return True
    except OSError:
        return False


st.set_page_config(page_title="AQI Prediction", page_icon="🌫️", layout="centered")

info = load_info()
model = load_model()
ranges = info["input_ranges"]
countries = info["countries"]

st.title("Global Air Pollution: AQI Prediction System")
st.write(
    "Enter the AQI values of three pollutants (CO, Ozone, NO2) measured in a city and choose its country. "
    "The model estimates the city's overall **Air Quality Index (AQI)** when its PM2.5 reading is not available."
)
st.info(
    "Educational project for AIC354 Machine Learning Fundamentals. The prediction is a rough statistical "
    "estimate (typical error about 30 AQI points), not an official air-quality measurement."
)

with st.form("city_form"):
    st.subheader("City air-quality readings")
    default_country = countries.index("Germany") if "Germany" in countries else 0
    country = st.selectbox("Country", options=countries, index=default_country)

    col1, col2, col3 = st.columns(3)
    with col1:
        co = st.number_input(f"CO AQI Value  [range {ranges['CO AQI Value'][0]} to {ranges['CO AQI Value'][1]}]",
                             min_value=ranges["CO AQI Value"][0], max_value=ranges["CO AQI Value"][1],
                             value=1, step=1)
    with col2:
        ozone = st.number_input(f"Ozone AQI Value  [range {ranges['Ozone AQI Value'][0]} to {ranges['Ozone AQI Value'][1]}]",
                                min_value=ranges["Ozone AQI Value"][0], max_value=ranges["Ozone AQI Value"][1],
                                value=30, step=1)
    with col3:
        no2 = st.number_input(f"NO2 AQI Value  [range {ranges['NO2 AQI Value'][0]} to {ranges['NO2 AQI Value'][1]}]",
                              min_value=ranges["NO2 AQI Value"][0], max_value=ranges["NO2 AQI Value"][1],
                              value=2, step=1)
    submitted = st.form_submit_button("Predict AQI")

if submitted:
    city = {"CO AQI Value": co, "Ozone AQI Value": ozone, "NO2 AQI Value": no2, "Country": country}
    frame = pd.DataFrame([city])[FEATURES]
    predicted = float(np.clip(model.predict(frame)[0], 0, 500))   # AQI is always between 0 and 500
    st.session_state["result"] = {"inputs": city, "aqi": predicted, "feedback_sent": False}

result = st.session_state.get("result")
if result:
    st.divider()
    st.subheader("Result")
    category, kind = aqi_category(result["aqi"])
    getattr(st, kind)(f"Predicted AQI: **{result['aqi']:.0f}**, category: **{category}**")

    col_a, col_b = st.columns(2)
    col_a.metric("Predicted AQI Value", f"{result['aqi']:.1f}")
    col_b.metric("Typical error (cross-validated RMSE)", f"± {info['cv_mean_rmse']:.1f}")
    st.caption(
        "AQI runs from 0 (clean air) to 500 (hazardous). The typical error is the average RMSE of the "
        "model over 10-fold cross validation, in AQI points. It is large because the PM2.5 reading, which "
        "usually decides the AQI, is not used."
    )

    entered = result["inputs"]
    st.table(pd.DataFrame({"Input": list(entered.keys()),
                           "Value": [str(v) for v in entered.values()]}))

    st.subheader("Feedback")
    if result["feedback_sent"]:
        st.success("Thank you for your feedback!")
    else:
        with st.form("feedback_form", clear_on_submit=True):
            rating = st.radio("Was this result easy to understand?", ["Yes", "Somewhat", "No"], horizontal=True)
            comment = st.text_area("Comments or suggestions (optional)", max_chars=300)
            send = st.form_submit_button("Send feedback")
        if send:
            save_feedback(rating, comment, entered, result["aqi"])
            result["feedback_sent"] = True
            st.success("Thank you for your feedback!")

with st.expander("About this model"):
    st.write(
        f"The model is **{info['model']}**, trained on {info['rows_used']:,} cities from the Global Air "
        "Pollution dataset. It was chosen as the best of 5 regression algorithms using "
        f"{info['cv_folds']}-fold cross validation."
    )
    st.write(
        f"- Cross-validated RMSE: **{info['cv_mean_rmse']:.2f}** AQI points (± {info['cv_std_rmse']:.2f} over the folds)\n"
        f"- Cross-validated MAE: **{info['cv_mean_mae']:.2f}** AQI points\n"
        f"- Cross-validated R2: **{info['cv_mean_r2']:.3f}**"
    )
    st.write(
        "By definition, a city's overall AQI is the highest of its individual pollutant AQI values, "
        "and PM2.5 is the highest one in about 78% of cities. PM2.5 was deliberately left out, so the model "
        "has to estimate the AQI from CO, Ozone, NO2 and the country; the result is a rough estimate. "
        "The model only accepts values inside the ranges seen in the training data."
    )
