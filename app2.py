# app2.py

import streamlit as st
import pandas as pd
import numpy as np
import json
from joblib import load
from counterfactuals_generator import generate_counterfactuals_for_query
import altair as alt

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM

# --- Page Config ---
st.set_page_config(page_title="Insurance Charge Predictor", layout="wide",initial_sidebar_state="collapsed")

# --- Custom CSS Styling ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=Poppins:wght@400;600;700&display=swap');

html, body {
    background-color: #f0f8ff;
    margin: 0;
    padding: 0;
}

.stApp {
    background: linear-gradient(180deg, #e3f2fd 0%, #bbdefb 100%);
    padding-top: 0;
    min-height: 100vh;
    font-family: 'Inter', sans-serif;
}

.header-container {
    background-color: #1e3a8a;
    padding: 2rem 2rem 4rem;
    text-align: center;
    color: white;
}

.header-container h1 {
    font-family: 'Poppins', sans-serif;
    font-size: 3rem;
    margin-bottom: 0.5rem;
}

.header-container p {
    font-size: 1.2rem;
    font-weight: 400;
    color: #cfd8dc;
}

.block-container {
    padding: 2rem 4rem;
}

.stTabs [data-baseweb="tab"] {
    font-size: 1.2rem;
    font-weight: 600;
}

.stTabs [data-baseweb="tab"][aria-selected="true"] {
    border-bottom: 4px solid #1e88e5;
    color: #1e88e5;
}

.stButton>button {
    background-color: #1e88e5;
    color: white;
    border-radius: 999px;
    padding: 0.5rem 1.5rem;
    font-weight: 600;
    border: none;
}

.stMetricLabel {
    color: #1e3a8a;
}
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.markdown("""
<div class='header-container'>
    <h1>🩺 Insurance Charge Predictor</h1>
    <p>Predict costs, explore insights, and generate counterfactuals with machine learning.</p>
</div>
""", unsafe_allow_html=True)

# --- Tabs ---
tabs = st.tabs(["🏠 Home", "📊 Insights", "🔮 Predictor", "📈 Model Analysis", "👤 About"])

# --- Shared resources ---
model_options = {
    "Linear Regression": "baseline_models/linear_regression_pipeline.pkl",
    "Random Forest": "tuned_models/random_forest_pipeline.pkl",
    "Gradient Boosting": "tuned_models/gradient_boosting_pipeline.pkl",
    "XGBoost": "tuned_models/xgboost_pipeline.pkl",
    "CatBoost": "tuned_models/catboost_pipeline.pkl"
}

features = ['age', 'sex', 'bmi', 'children', 'smoker', 'region']

with open("tuned_models/model_performance_summary.json") as f:
    performance_data = json.load(f)

# --- Home Tab ---
with tabs[0]:
    st.markdown("""
    # <span style="color:#0d47a1;">Medical Insurance Cost Estimator</span>
    
    <div style="border-left: 4px solid #0d47a1; padding-left: 15px; margin: 20px 0;">
    <p style="font-style: italic; color: #546e7a;">Precision healthcare financial planning powered by advanced analytics</p>
    </div>
    
    <hr style="margin: 25px 0;">
    
    ### <span style="color:#0d47a1;">Clinical-Grade Insurance Assessment</span>
    
    Our state-of-the-art platform enables healthcare professionals and patients to:
    
    * **Generate evidence-based cost projections** utilizing clinical and demographic parameters
    * **Perform comparative analysis** across multiple validated predictive models
    * **Examine counterfactual scenarios** to identify modifiable health determinants
    
    <div style="background-color: #e3f2fd; padding: 15px; border-radius: 5px; margin-top: 20px;">
    <p><strong>IMPORTANT:</strong> This tool is designed for informational purposes and should be used in consultation with healthcare financial advisors. Results represent statistical projections based on historical data.</p>
    </div>
    """, unsafe_allow_html=True)

# --- Insights Tab ---
with tabs[1]:
    st.header("Data Insights")
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

def explain_counterfactual(original_input: pd.DataFrame, counterfactuals: pd.DataFrame) -> str:
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)

    original = original_input.to_dict(orient='records')[0]
    changed_attributes = {}

    for idx, cf in counterfactuals.iterrows():
        for key in original.keys():
            if key in cf:
                orig_value = original[key]
                cf_value = cf[key]
                # Ignore tiny BMI changes
                if isinstance(orig_value, float) and isinstance(cf_value, float):
                    if abs(orig_value - cf_value) < 1.0:
                        continue
                if orig_value != cf_value:
                    changed_attributes[key] = changed_attributes.get(key, 0) + 1

    if not changed_attributes:
        change_summary = "No significant attribute changes detected."
    else:
        change_summary = ", ".join([f"{k}" for k in changed_attributes.keys()])

    prompt = f"""
You are an expert at explaining healthcare insurance predictions.

The original patient profile is:
{original}

The table shows alternative patient profiles that would lower insurance costs.

The most common changes in the alternatives are: {change_summary}.

Explain in plain English what the table shows about lowering insurance costs.
Focus on what actions (like quitting smoking, lowering BMI) help, and summarize briefly.
Keep your explanation under 80 words.
"""

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    output = model.generate(input_ids, max_new_tokens=100)
    explanation = tokenizer.decode(output[0], skip_special_tokens=True)
    return explanation

with tabs[2]:
    st.header("Predict Insurance Charge")
    model_choice = st.selectbox("Choose a model:", list(model_options.keys()))

    try:
        pipeline = load(model_options[model_choice])
        preprocessor = pipeline.named_steps['preprocessor']
    except Exception as e:
        st.error(f"❌ Could not load model: {e}")
        st.stop()

    st.subheader("Enter Patient Information")
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
            st.subheader("Counterfactual Explanations")
            try:
                cf_df = generate_counterfactuals_for_query(user_input, max_cost=max_cost)
                if not cf_df.empty:
                    st.markdown("Alternative patient profiles to lower the cost:")
                    st.dataframe(cf_df)
                    #with st.spinner("💬 Generating explanation..."):
                    #    explanation = explain_counterfactual(user_input, cf_df)
                    #    st.markdown("**🧠 AI Explanation:**")
                    #    st.markdown(explanation)
                else:
                    st.warning("⚠️ No counterfactuals found under the given cost constraint.")
            except Exception as e:
                st.error(f"❌ Error generating counterfactuals: {e}")
        else:
            st.info("ℹ️ Counterfactuals are only available for the **XGBoost** model.")

# --- Model Analysis Tab ---
with tabs[3]:
    st.header("Model Performance Analysis")
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

    st.subheader("Histogram of Predicted Charges")
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

    st.subheader("Residuals (Prediction Errors)")
    residual_chart = alt.Chart(df).mark_bar(opacity=0.7).encode(
        x=alt.X("residuals:Q", bin=alt.Bin(maxbins=30), title="Residual (Actual - Predicted)"),
        y=alt.Y("count():Q", title="Frequency")
    ).properties(width=700, height=400)
    st.altair_chart(residual_chart)

    perf = performance_data.get(model_choice_analysis, {})
    st.subheader("Model Performance Metrics")
    st.metric("RMSE (Root Mean Square Error)", f"${perf.get('rmse', 0):,.2f}")
    st.metric("R² Score", f"{perf.get('r2', 0):.3f}")

    r2 = perf.get('r2', 0)
    if r2 > 0.85:
        st.success("This model explains most of the variance in charges. It's highly predictive.")
    elif r2 > 0.65:
        st.info("This model explains a decent amount of variance but might miss some subtle patterns.")
    else:
        st.warning("This model struggles to explain variation in charges. Consider a better model or more features.")

# --- About Tab ---
with tabs[4]:
    st.header("About")
    st.markdown(""" 
    ### <span style="color:#0d47a1;">Our Platform</span>
    
    The Insurance Charge Predictor is a **user-friendly web application** designed to help users estimate healthcare insurance costs based on demographic and lifestyle information. With a clean, modern interface and intuitive navigation, the app guides users through predicting insurance charges, exploring **AI-generated counterfactual profiles**, and analyzing model performance across different machine learning algorithms.
    
    Using fine-tuned models like **XGBoost and Linear Regression**, the website offers not only **accurate predictions** but also **interpretable insights**—such as how changes in inputs (e.g., quitting smoking or lowering BMI) could reduce costs. The app’s counterfactual generator provides personalized "what-if" scenarios, enhancing decision-making power for patients and consumers.
    
    The tool is fully functional in its current state and built for scalability via Streamlit, with backend support for additional models and datasets. It is well-documented and includes clear instructions and modular design for further development or deployment in clinical or financial settings.
    
    <div style="background-color: #e3f2fd; padding: 15px; border-radius: 5px; margin-top: 20px;">
    <strong>Why It Matters:</strong> 
    <ul>
        <li>📈 <strong>Impact:</strong> Empowers users to make cost-saving health decisions using predictive modeling.</li>
        <li>🧠 <strong>AI Integration:</strong> Leverages advanced ML techniques with explainability.</li>
        <li>🖥️ <strong>Design & Usability:</strong> Modern, responsive interface with smooth interaction.</li>
        <li>📊 <strong>Insights:</strong> Visual analytics and model comparisons for deeper understanding.</li>
        <li>🚀 <strong>Scalability:</strong> Easily extendable and deployable for wider use cases.</li>
    </ul>
    </div>
                
    ### <span style="color:#0d47a1;">Developers</span>
    **Allison Xin:** Caltech UG2, Computer Science + Economics
    
    **Clara Yu:** Caltech UG2, Computation and Neural Systems
    
    ### <span style="color:#0d47a1;">Sources</span>
    Kaggle Dataset: https://www.kaggle.com/datasets/teertha/ushealthinsurancedataset?resource=download
    
    Libraries/Packages: Sklearn, XGBoost, Catboost, Pandas, Numpy, Streamlit, Altair    
                
    
    
    """, unsafe_allow_html=True)
