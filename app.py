import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from counterfactuals_generator import generate_counterfactuals_for_query

# --- Page Setup ---
st.set_page_config(page_title="Insurance Charge Predictor", page_icon="💡", layout="centered")
st.markdown("<h1 style='text-align: center;'>💡 Insurance Charge Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray;'>Estimate costs and explore counterfactuals</p>", unsafe_allow_html=True)
st.markdown("---")

# --- Model Selection ---
model_paths = {
    "Linear Regression": "baseline_models/linear_regression_pipeline.pkl",
    "Random Forest": "baseline_models/random_forest_pipeline.pkl",
    "Gradient Boosting": "baseline_models/gradient_boosting_pipeline.pkl",
    "XGBoost (supports counterfactuals)": "baseline_models/xgboost_pipeline.pkl",
    "CatBoost": "baseline_models/catboost_pipeline.pkl"
}

with st.sidebar:
    st.header("🔧 Model Settings")
    model_choice = st.selectbox("Choose a Model:", list(model_paths.keys()))
    pipeline_path = model_paths[model_choice]
    try:
        pipeline = load(pipeline_path)
    except Exception as e:
        st.error(f"❌ Could not load model: {e}")
        st.stop()

# --- Input Form ---
st.subheader("📋 Enter Patient Info")
with st.form("predict_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 65, 30)
        sex = st.selectbox("Sex", ["male", "female"])
        bmi = st.slider("BMI", 10.0, 50.0, 25.0)
    with col2:
        children = st.slider("Children", 0, 5, 0)
        smoker = st.selectbox("Smoker", ["yes", "no"])
        region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])
    submitted = st.form_submit_button("📊 Predict")

if submitted:
    user_input = pd.DataFrame([{
        'age': age,
        'sex': sex,
        'bmi': bmi,
        'children': children,
        'smoker': smoker,
        'region': region
    }])

    # --- Prediction Display ---
    pred = pipeline.predict(user_input)[0]
    st.markdown("### 💰 Predicted Charge")
    st.metric(label="Estimated Cost", value=f"${pred:,.2f}", delta=None)

    # --- Counterfactual Display ---
    if "XGBoost" in model_choice:
        st.markdown("---")
        st.subheader("🧠 Counterfactual Explanations (XGBoost only)")
        cf_df = generate_counterfactuals_for_query(user_input)
        if not cf_df.empty:
            st.markdown("✅ **Suggestions for lowering cost:**")
            st.dataframe(cf_df, use_container_width=True)
        else:
            st.warning("⚠️ No counterfactuals could be generated.")
    else:
        st.info("ℹ️ Counterfactuals are only available for the **XGBoost** model.")

# --- Footer ---
st.markdown("---")
st.markdown(
    "<p style='text-align: center; font-size: 0.9em; color: gray;'>Built with ❤️ for exploring healthcare cost prediction and fairness.</p>",
    unsafe_allow_html=True
)
