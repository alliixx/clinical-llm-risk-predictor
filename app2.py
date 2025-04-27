# app2.py

import streamlit as st
import pandas as pd
import numpy as np
import json
from joblib import load
from counterfactuals_generator import generate_counterfactuals_for_query
import altair as alt
import time
from streamlit_lottie import st_lottie
import json

import google.generativeai as genai
import openai
from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv("key.env")
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
#genai.configure(api_key="AIzaSyDS8OBWXBhUGpdNm5-BzvsCmJMTEzXXvc8")

st.set_page_config(
    page_title="Insurance Charge Predictor",
    layout="wide",
    initial_sidebar_state="collapsed"
)

def load_lottie(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Load a Lottie JSON animation
lottie_animation = load_lottie("splash.json")  # Make sure you have a Lottie JSON file!

# --- Splash Screen ---
if "splash_shown" not in st.session_state:
    st.session_state.splash_shown = False

if not st.session_state.splash_shown:
    # Create three columns with the middle one containing the animation
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st_lottie(
            lottie_animation,
            speed=1,
            reverse=False,
            loop=True,
            quality="high",
            height=600,
            width=600
        )
        st.title("Loading Insurance Charge Predictor...")
    
    time.sleep(3)  # Wait for 3 seconds
    st.session_state.splash_shown = True
    st.rerun() 
else:
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
    <div style="
        background-image: url('https://wallpapers.com/images/hd/minimalist-blue-6j2n0kdmvlxc1kv2.jpg');
        background-size: cover;
        background-position: center 70%;
        padding: 80px;
        border-radius: 10px;
        color: white;
        text-align: center;
    ">
        <div style="
            display: inline-block;
            padding: 20px 40px;
            border: 2px solid white;
            border-radius: 10px;
            background-color: rgba(0,0,0,0.4);
        ">
            <h1>🩺 Insurance Charge Predictor</h1>
            <p>Predict costs, explore insights, and generate counterfactuals with machine learning.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)


    # --- Tabs ---
    tabs = st.tabs(["🏠 Home", "💵 Insurance 101", "📊 Insights", "🔮 Predictor", "📈 Model Analysis", "👤 About"])

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
        # <span style="color:#000000;">Medical Insurance Cost Estimator</span>
        
        <div style="border-left: 4px solid #000000; padding-left: 15px; margin: 20px 0;">
        <p style="font-style: italic; color: #546e7a;">Transparent precision healthcare financial planning powered by advanced analytics</p>
        </div>
        
        <hr style="margin: 25px 0;">
        
        ### <span style="color:#000000;">Clinical-Grade Insurance Assessment</span>
        
        Our state-of-the-art platform enables healthcare professionals and patients to:
        
        * **Generate evidence-based cost projections** utilizing clinical and demographic parameters
        * **Perform comparative analysis** across multiple validated predictive models
        * **Examine counterfactual scenarios** to identify modifiable health determinants
                    
        <div style="background-color: #e3f2fd; padding: 15px; border-radius: 5px; margin-top: 20px;">
        <strong>Did you know?</strong> 
        <ul>
        <li><strong>Widespread Non-Compliance with Transparency Rules:</strong>  
        Despite the federal Hospital Price Transparency Rule effective since January 2021, only <strong>21.1% of hospitals</strong> reviewed were fully compliant as of November 2024. This lack of transparency hinders patients' ability to make informed financial decisions.  
        (<a href="https://www.patientrightsadvocate.org/pra-reports" target="_blank">Patient Rights Advocate Report</a>)
        </li>
        <br>
        <li><strong>Significant Price Variability for Identical Services:</strong>  
        Within the same metropolitan areas, prices for identical healthcare services can vary by <strong>40% to 50%</strong>. Such disparities make it difficult for patients to predict costs and can result in unexpected medical bills.  
        (<a href="https://www.mckinsey.com/industries/healthcare/our-insights/how-price-transparency-could-affect-us-healthcare-markets" target="_blank">McKinsey Report</a>)
        </li>
        <br>
        <li><strong>Potential for Substantial Cost Savings:</strong>  
        Effective transparency measures could reduce national healthcare spending by more than <strong>$1 trillion annually</strong> while improving health outcomes. Transparent information empowers consumers to make cost-effective healthcare choices.  
        (<a href="https://www.patientrightsadvocate.org/pra-reports" target="_blank">Patient Rights Advocate Report</a>)
        </li>
        <br>
        <li><strong>Enhanced Patient Engagement and Decision-Making:</strong>  
        Transparent pricing helps patients better predict out-of-pocket costs, reduces the likelihood of surprise bills, and supports more proactive and informed healthcare decision-making.  
        (<a href="https://journalofethics.ama-assn.org/article/how-might-patients-and-physicians-use-transparent-health-care-prices-guide-decisions-and-improve/2022-11" target="_blank">AMA Journal of Ethics</a>)
        </li>
        </ul>
        """, unsafe_allow_html=True)

        st.markdown("""
        <p><strong>IMPORTANT:</strong> This tool is designed for informational purposes and should be used in consultation with healthcare financial advisors. Results represent statistical projections based on historical data.</p>
        """, unsafe_allow_html=True)

    # --- Insurance 101 Tab ---
    with tabs[1]:
        st.markdown("## 💬 Ask Our AI Chatbox About Health Insurance")

        user_question = st.text_input("Type your question:")

        if user_question:
            with st.spinner("Thinking..."):
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a helpful expert in health insurance. Explain health insurance concepts simply, clearly, and accurately."},
                        {"role": "user", "content": user_question}
                    ],
                    temperature=0.5,
                    max_tokens=300
                )
                
                answer = response.choices[0].message.content
                st.success(answer)


        st.markdown("""
        # 💡 Understanding Key Health Insurance Terms

        <hr>

        ## 💰 Health Insurance Premiums

        - **Definition**: A premium is the fixed monthly payment you make to maintain your health insurance coverage, regardless of whether you use medical services that month.
        - **Key Point**: Premiums are separate from other out-of-pocket costs like deductibles and copays. They do not count toward your deductible.  
        (<a href="https://www.verywellhealth.com/do-premiums-count-toward-your-deductible-1738443" target="_blank">VeryWell Health</a>)

        <hr>

        ## 💳 Copayments (Copays)

        - **Definition**: A copay is a predetermined, fixed amount you pay for specific healthcare services, such as $20 for a doctor's visit or $10 for a prescription.
        - **Key Point**: Copays are due at the time of service and can vary depending on the type of service or specialist.  
        (<a href="https://www.investopedia.com/terms/c/copay.asp" target="_blank">Investopedia</a>)

        <hr>

        ## 🧾 Deductibles

        - **Definition**: A deductible is the amount you pay out-of-pocket for healthcare services before your insurance begins to cover costs. For example, with a $1,000 deductible, you pay the first $1,000 of covered services yourself.
        - **Key Point**: After meeting your deductible, you typically pay a portion of costs through copays or coinsurance, while your insurer covers the rest.  
        (<a href="https://www.aetna.com/health-guide/explaining-premiums-deductibles-coinsurance-and-copays.html" target="_blank">Aetna</a>)

        <hr>

        ## 🔄 Coinsurance

        - **Definition**: Coinsurance is the percentage of costs you share with your insurance company after meeting your deductible. For instance, with a 20% coinsurance, you pay 20% of the service cost, and your insurer pays 80%.
        - **Key Point**: Coinsurance amounts can vary based on the service and whether the provider is in-network.  
        (<a href="https://www.cigna.com/knowledge-center/copays-deductibles-coinsurance" target="_blank">Cigna</a>)

        <hr>

        ## 🏥 HMO vs. PPO Plans

        <table>
        <thead>
        <tr>
        <th style="text-align:left;">Feature</th>
        <th style="text-align:left;">HMO (Health Maintenance Organization)</th>
        <th style="text-align:left;">PPO (Preferred Provider Organization)</th>
        </tr>
        </thead>
        <tbody>
        <tr>
        <td><strong>Primary Care Physician</strong></td>
        <td>Required; you must choose a PCP who coordinates all your care and provides referrals to specialists.  
        (<a href="https://www.webmd.com/health-insurance/hmo-vs-ppo" target="_blank">WebMD</a>)</td>
        <td>Not required; you can see specialists without referrals.  
        (<a href="https://www.webmd.com/health-insurance/hmo-vs-ppo" target="_blank">WebMD</a>)</td>
        </tr>
        <tr>
        <td><strong>Network Restrictions</strong></td>
        <td>Must use in-network providers for all non-emergency services.</td>
        <td>Flexibility to see both in-network and out-of-network providers, though out-of-network care usually costs more.</td>
        </tr>
        <tr>
        <td><strong>Costs</strong></td>
        <td>Generally lower premiums and out-of-pocket costs.</td>
        <td>Typically higher premiums and out-of-pocket costs for greater flexibility.</td>
        </tr>
        <tr>
        <td><strong>Best For</strong></td>
        <td>Individuals seeking lower costs and willing to work within a managed network.</td>
        <td>Individuals desiring more provider options and willing to pay higher costs for that flexibility.</td>
        </tr>
        </tbody>
        </table>

        <hr>

        Understanding these terms can help you make informed decisions when selecting a health insurance plan that best fits your healthcare needs and financial situation.
        """, unsafe_allow_html=True)

        st.markdown("""
        ### 📸 Healthcare Price Transparency

        <img src="https://mma.prnewswire.com/media/1863473/AKASA_Healthcare_Price_Transparency.jpg?p=twitter" alt="Healthcare Price Transparency" width="700">
        """, unsafe_allow_html=True)

        st.markdown("""
        # 🩺 Understanding Health Insurance Premiums

        **Health insurance premiums** are the monthly payments individuals or employers make to maintain health coverage. These costs can vary widely based on factors like age, location, plan type, and insurer. Over the past decade, premiums have consistently outpaced wage growth, placing a significant financial burden on many Americans.

        <hr>

        ## 📈 Current Landscape and Challenges

        <ul>
        <li><strong>Rising Premiums:</strong>  
        In 2025, health insurance premiums on the Health Insurance Marketplace are increasing in 35 states and Washington, D.C., with Vermont, Alaska, and North Dakota experiencing the highest hikes. For instance, Vermont's lowest-cost silver plan is rising from $948 to $1,275 monthly.  
        (<a href="https://www.investopedia.com/health-insurance-premiums-are-rising-in-35-states-california-new-york-texas-florida-is-yours-one-of-them-8743374" target="_blank">Investopedia</a>)
        </li><br>

        <li><strong>Geographic Variability:</strong>  
        Premium costs can differ dramatically between states and even within regions. Factors such as local healthcare costs, competition among insurers, and state regulations contribute to this variability.  
        (<a href="https://www.commonwealthfund.org/publications/issue-briefs/2024/dec/trends-employer-health-insurance-costs-2014-2023" target="_blank">Commonwealth Fund</a>)
        </li><br>

        <li><strong>Lack of Transparency:</strong>  
        Despite federal efforts like the Hospital Price Transparency Rule, many healthcare providers and insurers have been slow to disclose pricing information. This opacity makes it challenging for consumers to compare plans and anticipate out-of-pocket expenses.  
        (<a href="https://jamanetwork.com/journals/jama-health-forum/fullarticle/2811063" target="_blank">JAMA Network</a>)
        </li>
        </ul>

        <hr>

        ## 🔍 The Need for Greater Transparency

        <ul>
        <li><strong>Empowering Consumers:</strong>  
        Transparent pricing allows individuals to make informed decisions about their healthcare, potentially leading to cost savings and better health outcomes.
        </li><br>

        <li><strong>Market Efficiency:</strong>  
        When prices are clear, competition among providers can drive down costs, benefiting consumers and the broader healthcare system.
        </li><br>

        <li><strong>Policy Initiatives:</strong>  
        Recent regulations, such as the Transparency in Coverage rule, aim to shed light on negotiated rates between insurers and providers. However, the effectiveness of these measures depends on consistent enforcement and public accessibility of the data.  
        (<a href="https://www.cms.gov/priorities/key-initiatives/healthplan-price-transparency" target="_blank">CMS.gov</a>)
        </li>
        </ul>

        <hr>

        """, unsafe_allow_html=True)


    # --- Insights Tab ---
    with tabs[2]:
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

        st.markdown("""
        **Key Insight:**  
        - Most insurance charges fall between 1,000 and 12,000 dollars.  
        - A few extreme outliers suggest high-cost medical cases such as surgeries or chronic illnesses.
        """)

        st.subheader("BMI vs Charges by Smoking Status")
        scatter_chart = alt.Chart(df).mark_circle(size=60, opacity=0.4).encode(
            x=alt.X('bmi', title='BMI'),
            y=alt.Y('charges', title='Insurance Charges ($)'),
            color='smoker:N',  # Color by smoker yes/no
            tooltip=['bmi', 'charges', 'smoker']
        ).properties(width=1100, height=400).interactive()
        st.altair_chart(scatter_chart)

        st.markdown("""
        **Key Insight:**  
        - Smokers consistently face higher insurance charges across all BMI ranges.  
        - Higher BMI is also associated with rising charges, particularly among smokers.
        """)

        st.subheader("Average Charges by Region")
        region_avg = df.groupby('region')['charges'].mean().reset_index()
        st.line_chart(region_avg.rename(columns={'charges': 'Average Charges'}).set_index('region'))

        st.markdown("""
        **Key Insight:**  
        - Average charges vary moderately across regions.  
        - The Southeast region shows slightly higher costs on average.
        """)

        st.subheader("Average Charges by Smoking Status")
        smoker_avg = df.groupby('smoker')['charges'].mean().reset_index()
        st.bar_chart(smoker_avg.rename(columns={'charges': 'Average Charges'}).set_index('smoker'))

        st.markdown("""
        **Key Insight:**  
        - Smokers pay nearly triple the insurance charges compared to non-smokers on average.  
        - Smoking status is a critical factor influencing insurance costs.
        """)

        st.subheader("Interactive Correlation Heatmap")

        import plotly.express as px

        corr = df.corr(numeric_only=True)

        fig = px.imshow(
            corr,
            text_auto=True,       # show correlation numbers inside
            color_continuous_scale='RdBu_r',  # Red to Blue reversed
            aspect="auto",         # makes it compact automatically
            width=600,             # total width of plot
            height=400             # total height
        )

        fig.update_layout(
            margin=dict(l=30, r=30, t=30, b=30),
            coloraxis_colorbar=dict(title="Correlation")
        )

        st.plotly_chart(fig, use_container_width=False)


        st.markdown("""
        **Key Insight:**  
        - Smoking, age, and BMI are the strongest positive correlates with higher insurance charges.  
        - Number of children has little correlation with costs.
        """)

        st.subheader("📄 Download Dataset")
        st.download_button(
            label="Download Insurance Data as CSV",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name='insurance_data.csv',
            mime='text/csv'
        )

        # Final Insights Summary
        st.markdown("""
        ---
        ### 🔎 Summary of Key Findings
        - **Smoking** has the strongest impact on raising insurance charges.
        - **Higher BMI** contributes significantly to increased costs, especially among smokers.
        - **Regional differences** exist but are less dramatic.
        - **Transparency** into these factors helps users make informed health and financial decisions.
        """)

    # --- Predictor Tab ---

    def openai_generate_counterfactual_explanation(original_profile, counterfactual_profiles):
        prompt = f"""
    You are an expert in healthcare insurance analysis.

    The original patient profile is:
    {original_profile}

    The following alternative profiles could lead to reduced insurance costs:
    {counterfactual_profiles}

    Explain in simple terms what changes in the patient attributes contribute to lower insurance charges. Focus on actionable factors such as smoking status, BMI, and region.
    """
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content


    def generate_counterfactual_explanation(original_profile, counterfactual_profiles):
        prompt = f"""
    You are an expert in healthcare insurance analysis.

    The original patient profile is:
    {original_profile}

    The following alternative profiles could lead to reduced insurance costs:
    {counterfactual_profiles}

    Explain in simple terms what changes in the patient attributes contribute to lower insurance charges. Focus on actionable factors such as smoking status, BMI, and region.
    """
        model = genai.GenerativeModel("gemini-2.5-pro-exp-03-25")
        response = model.generate_content(prompt)
        return response.text

    with tabs[3]:
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
                        with st.spinner("Generating explanation with AI..."):
                            try:
                                explanation = openai_generate_counterfactual_explanation(user_input, cf_df.to_dict(orient="records"))
                                st.subheader("🧠 AI-Generated Explanation:")
                                st.success(explanation)
                            except Exception as e:
                                st.error(f"❌ Error generating explanation: {e}")
                    else:
                        st.warning("⚠️ No counterfactuals found under the given cost constraint.")
                except Exception as e:
                    st.error(f"❌ Error generating counterfactuals: {e}")
            else:
                st.info("ℹ️ Counterfactuals are only available for the **XGBoost** model.")

    # --- Model Analysis Tab ---
    with tabs[4]:
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
    with tabs[5]:
        st.header("About")
        st.markdown(""" 
        ### <span style="color:#000000;">Our Platform</span>
        
        The **Insurance Charge Predictor** is a **user-friendly web application** designed to help users estimate healthcare insurance costs based on demographic and lifestyle information. With a clean, modern interface and intuitive navigation, the app guides users through predicting insurance charges, exploring **AI-generated counterfactual profiles**, and analyzing model performance across different machine learning algorithms.
        
        Utilizing fine-tuned models like **XGBoost** and **Linear Regression**, the platform offers not only **accurate predictions** but also **interpretable insights** — such as how changes in inputs (e.g., quitting smoking or lowering BMI) could reduce costs. The app’s counterfactual generator provides personalized "what-if" scenarios, enhancing decision-making power for patients and consumers.
        
        A standout feature is the integration of **AI-generated explanations** for counterfactuals. By leveraging advanced language models, the app translates complex data into clear, actionable narratives, helping users understand how specific changes in their profiles could impact insurance costs.
        
        The tool is fully functional and built for scalability via Streamlit, with backend support for additional models and datasets. It is modularly designed for easy future development and deployment in clinical or financial settings.
        
        <div style="background-color: #e3f2fd; padding: 15px; border-radius: 5px; margin-top: 20px;">
        <strong>Why It Matters:</strong> 
        <ul>
            <li>📈 <strong>Impact:</strong> Empowers users to make cost-saving healthcare decisions with predictive modeling.</li>
            <li>🧠 <strong>AI Integration:</strong> Leverages machine learning and generative AI for explainable and actionable outputs.</li>
            <li>🖥️ <strong>Design & Usability:</strong> Modern, responsive interface with seamless user experience.</li>
            <li>📊 <strong>Insights:</strong> Visual analytics and counterfactual explanations for deeper healthcare transparency.</li>
            <li>🚀 <strong>Scalability:</strong> Built for extension to broader populations and additional healthcare domains.</li>
        </ul>
        </div>
        
        ### <span style="color:#000000;">Addressing Health Insurance Disparities</span>
        
        In the United States, significant disparities persist in healthcare insurance access and costs, especially among racial minorities, low-income populations, and rural communities. Barriers such as socioeconomic status, systemic bias, and geographic inequities contribute to worse health outcomes and limited healthcare access ([source](https://www.ncbi.nlm.nih.gov/books/NBK425844/?utm_source=chatgpt.com)).
        
        Our platform advances **healthcare transparency** by providing clear, personalized estimates of insurance charges and highlighting how lifestyle or demographic factors can impact costs. By making insurance pricing more understandable and accessible, we empower individuals — particularly those historically underserved — to advocate for fairer healthcare coverage and take proactive steps toward cost reduction.
        
        Through AI-driven counterfactual explanations, our project also demystifies the complex world of healthcare pricing, promoting greater equity, informed decision-making, and awareness of structural healthcare disparities.
        
        ### <span style="color:#000000;">Developers</span>
        
        **Allison Xin:** Caltech UG2, Computer Science + Economics  
        **Clara Yu:** Caltech UG2, Computation and Neural Systems
        
        ### <span style="color:#000000;">Sources</span>
        
        Kaggle Dataset: [US Health Insurance Dataset](https://www.kaggle.com/datasets/teertha/ushealthinsurancedataset?resource=download)  
        Libraries/Packages: Sklearn, XGBoost, Catboost, Pandas, Numpy, Streamlit, Altair, OpenAI, Lottie
        
        """, unsafe_allow_html=True)
