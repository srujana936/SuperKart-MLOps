
from pathlib import Path
import json

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "superkart_best_model.joblib"
SCHEMA_PATH = ROOT / "models" / "feature_schema.json"

st.set_page_config(
    page_title="SuperKart Sales Prediction",
    page_icon="📊",
    layout="centered"
)

st.title("SuperKart Sales Prediction")
st.write(
    "Enter product and store details to estimate "
    "Product Store Sales Total."
)

@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    with open(SCHEMA_PATH, "r") as file:
        schema = json.load(file)
    return model, schema

if not MODEL_PATH.exists() or not SCHEMA_PATH.exists():
    st.error(
        "Model files are missing. Run the training pipeline "
        "and ensure the model artifacts are committed to GitHub."
    )
    st.stop()

model, schema = load_artifacts()

with st.form("prediction_form"):
    st.subheader("Product details")

    input_values = {}

    for col in schema["numeric_columns"]:
        default_value = float(schema["numeric_defaults"].get(col, 0.0))
        input_values[col] = st.number_input(
            col.replace("_", " "),
            value=default_value,
            format="%.4f"
        )

    st.subheader("Product and store categories")

    for col in schema["categorical_columns"]:
        default_value = str(
            schema["categorical_defaults"].get(col, "")
        )
        input_values[col] = st.text_input(
            col.replace("_", " "),
            value=default_value
        )

    submitted = st.form_submit_button("Predict sales")

if submitted:
    # Keep feature names and ordering aligned with training data.
    input_df = pd.DataFrame([input_values])
    input_df = input_df[
        schema["numeric_columns"] + schema["categorical_columns"]
    ]

    try:
        prediction = float(model.predict(input_df)[0])

        st.success("Prediction completed")
        st.metric(
            "Predicted Product Store Sales Total",
            f"{prediction:,.2f}"
        )

        st.caption(
            "This is a model estimate, not a guaranteed sales outcome."
        )
        with st.expander("Review submitted inputs"):
            st.dataframe(input_df, use_container_width=True)

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
