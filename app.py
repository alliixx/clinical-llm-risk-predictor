# app.py (with animated background and enhanced UI)

import streamlit as st
import pandas as pd
import numpy as np
import json
from joblib import load
from counterfactuals_generator import generate_counterfactuals_for_query
import altair as alt

# --- Page Config ---
st.set_page_config(page_title="Insurance Charge Predictor", layout="wide")

# --- Custom CSS Styling ---
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #a1c4fd, #c2e9fb, #d4fc79);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        background-attachment: fixed;
        background-repeat: no-repeat;
        background-position: center;
    }

    @keyframes gradient {
        0% {
            background-position: 0% 50%;
        }
        50% {
            background-position: 100% 50%;
        }
        100% {
            background-position: 0% 50%;
        }
    }
    .main-title {
        font-size: 36px;
        font-weight: 700;
        color: #1a237e;
        text-align: center;
        padding-top: 10px;
    }
    .block-container {
        padding-top: 2rem;
    }
    .stButton>button {
        background-color: #3949ab;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5em 1em;
        border: none;
    }
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        color: #1a73e8 !important;
        border-bottom: 4px solid #1a73e8 !important;
        font-weight: bold;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: #1a1a1a !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("<div class='main-title'>💡 Insurance Charge Predictor</div>", unsafe_allow_html=True)

# --- Tabs ---
tabs = st.tabs(["🏠 Home", "📊 Insights", "🔮 Predictor", "📈 Model Analysis"])

# --- Shared resources ---
model_options = {
    "Linear Regression": "baseline_models/linear_regression_pipeline.pkl",
    "Random Forest": "tuned_models/random_forest_pipeline.pkl",
    "Gradient Boosting": "tuned_models/gradient_boosting_pipeline.pkl",
    "XGBoost": "tuned_models/xgboost_pipeline.pkl",
    "CatBoost": "tuned_models/catboost_pipeline.pkl"
}

features = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']

# Load performance summary
with open("tuned_models/model_performance_summary.json") as f:
    performance_data = json.load(f)

# --- Home Tab ---
with tabs[0]:
    st.markdown("""
    ## 👋 Welcome to the Insurance Charge Predictor
    This app empowers you to:
    - 🔮 Predict insurance costs based on lifestyle and demographic inputs
    - 🧠 Explore AI-generated counterfactual profiles (XGBoost only)
    - 📈 Analyze performance across machine learning models
    """)

# --- Insights Tab ---
with tabs[1]:
    st.markdown("## 📊 Data Insights")
    df = pd.read_csv("insurance.csv")

    st.subheader("Distribution of Charges")
    hist_values, hist_bins = np.histogram(df['charges'], bins=30)
    rounded_bins = np.round(hist_bins[:-1], -2)
    hist_df = pd.DataFrame({
        'Charge Bin Start ($)': rounded_bins,
        'Count': hist_values
    })
    st.bar_chart(hist_df.set_index('Charge Bin Start ($)'))

    st.subheader("BMI vs Charges")
    scatter_chart = alt.Chart(df).mark_circle(size=60, opacity=0.4).encode(
        x=alt.X('bmi', title='BMI'),
        y=alt.Y('charges', title='Insurance Charges ($)'),
        tooltip=['bmi', 'charges']
    ).properties(width=1100, height=400).interactive()
    st.altair_chart(scatter_chart)

    st.subheader("Average Charges by Region")
    region_avg = df.groupby('region')['charges'].mean().reset_index()
    st.line_chart(region_avg.rename(columns={'charges': 'Average Charges'}).set_index('region'))

# --- Predictor Tab ---
with tabs[2]:
    st.markdown("### 🔧 Model Settings")
    model_choice = st.selectbox("Choose a model:", list(model_options.keys()))

    try:
        pipeline = load(model_options[model_choice])
        preprocessor = pipeline.named_steps['preprocessor']
    except Exception as e:
        st.error(f"❌ Could not load model: {e}")
        st.stop()

    st.markdown("### 📋 Enter Patient Information")
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

        st.markdown("---")
        submit_col1, submit_col2 = st.columns([2, 1])
        with submit_col1:
            max_cost = st.number_input("🎯 Max Insurance Cost (XGBoost only)", min_value=1000, max_value=50000, value=10000, step=500) if model_choice == "XGBoost" else None
        with submit_col2:
            submitted = st.form_submit_button("🔮 Predict")

    if submitted:
        user_input = pd.DataFrame([{ 'age': age, 'sex': sex, 'bmi': bmi, 'children': children, 'smoker': smoker, 'region': region }])
        pred = pipeline.predict(user_input)[0]
        st.success(f"💰 **Predicted Insurance Charge:** ${pred:,.2f}")

        if model_choice == "XGBoost":
            st.markdown("### 🧠 Counterfactual Explanations")
            try:
                cf_df = generate_counterfactuals_for_query(user_input, max_cost=max_cost)
                if not cf_df.empty:
                    st.markdown("Here are some alternative patient profiles that could lower the predicted cost:")
                    st.dataframe(cf_df)
                else:
                    st.warning("⚠️ No counterfactuals found under the given cost constraint.")
            except Exception as e:
                st.error(f"❌ Error generating counterfactuals: {e}")
        else:
            st.info("ℹ️ Counterfactuals are only available for the **XGBoost** model.")

# --- Model Analysis Tab ---
with tabs[3]:
    st.markdown("### 🧪 Model Analysis")
    model_choice_analysis = st.selectbox("Choose a model to analyze:", list(model_options.keys()), key="model_analysis")

    try:
        pipeline_analysis = load(model_options[model_choice_analysis])
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        st.stop()

    df = pd.read_csv("insurance.csv")
    X = df[features]
    y = df['charges']
    y_pred = pipeline_analysis.predict(X)
    df['predicted'] = y_pred
    df['residuals'] = df['charges'] - df['predicted']

    st.markdown("#### 🔢 Histogram of Predicted Charges")
    bins = np.histogram_bin_edges(y_pred, bins=20)
    bin_labels = [f"${int(bins[i])} - ${int(bins[i+1])}" for i in range(len(bins)-1)]
    df['pred_bin'] = pd.cut(y_pred, bins=bins, labels=bin_labels)
    bin_counts = df['pred_bin'].value_counts().sort_index()
    chart_df = pd.DataFrame({
        "Charge Range": bin_counts.index.astype(str),
        "Count": bin_counts.values
    })
    chart = alt.Chart(chart_df).mark_bar().encode(
        x=alt.X("Charge Range:N", title="Predicted Charge Range"),
        y=alt.Y("Count:Q", title="Number of Predictions"),
        tooltip=["Charge Range", "Count"]
    ).properties(width=700, height=400)
    st.altair_chart(chart)

    st.markdown("#### 📉 Residuals (Prediction Errors)")
    residual_chart = alt.Chart(df).mark_bar(opacity=0.7).encode(
        x=alt.X("residuals:Q", bin=alt.Bin(maxbins=30), title="Residual (Actual - Predicted)"),
        y=alt.Y("count():Q", title="Frequency")
    ).properties(width=700, height=400)
    st.altair_chart(residual_chart)
    st.caption("Residuals show how far off the model's predictions are. A good model has residuals centered around 0.")

    perf = performance_data.get(model_choice_analysis, {})
    st.markdown("#### 📊 Model Performance")
    st.metric("RMSE (Root Mean Square Error)", f"${perf.get('rmse', 0):,.2f}")
    st.metric("R² Score", f"{perf.get('r2', 0):.3f}")

    st.markdown("#### 🧠 Interpretation")
    r2 = perf.get('r2', 0)
    if r2 > 0.85:
        st.success("This model explains most of the variance in charges. It's highly predictive.")
    elif r2 > 0.65:
        st.info("This model explains a decent amount of variance but might miss some subtle patterns.")
    else:
        st.warning("This model struggles to explain variation in charges. Consider a better model or more features.")