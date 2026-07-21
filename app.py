import os
import pickle
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from supabase import create_client

st.set_page_config(page_title="Healthcare Analytics & Risk Pipeline", layout="wide", page_icon="🏥")

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

@st.cache_resource(ttl=3600)
def init_supabase():
    if SUPABASE_URL and SUPABASE_KEY:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    return None

supabase = init_supabase()

@st.cache_data(ttl=3600)
def fetch_fda_data():
    if supabase:
        return supabase.table("fda_adverse_events").select("*").execute().data
    return []

@st.cache_data(ttl=3600)
def fetch_disease_data():
    if supabase:
        return supabase.table("disease_daily_metrics").select("*").execute().data
    return []

@st.cache_resource
def load_ml_model():
    model_path = 'healthcare_model.pkl'
    if os.path.exists(model_path):
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    return None

model_metadata = load_ml_model()

st.title("🏥 Real-Time Healthcare Analytics & Machine Learning Pipeline")
st.markdown("Live Data Ingestion from **Supabase Cloud Database** matched with an Optimized **XGBoost Inference Engine**.")
st.markdown("---")

if supabase is None:
    st.error("❌ Supabase Connection Failed! Please check your `.env` file.")
else:
    tab1, tab2 = st.tabs(["📊 Executive Analytics Dashboard", "🔮 Patient Risk Inference Portal"])
    
    with tab1:
        st.subheader("📡 Live Cloud Infrastructure Analytics")
        
        try:
            df_fda = pd.DataFrame(fetch_fda_data())
            df_disease = pd.DataFrame(fetch_disease_data())
            
            kpi1, kpi2, kpi3 = st.columns(3)
            with kpi1:
                st.metric("Total FDA Adverse Case Records", len(df_fda))
            with kpi2:
                high_risk_count = len(df_fda[df_fda['risk_level'] == 1]) if not df_fda.empty else 0
                pct = (high_risk_count / len(df_fda) * 100) if len(df_fda) > 0 else 0
                st.metric("High Risk Cases Flagged", high_risk_count, delta=f"{pct:.1f}% of Total")
            with kpi3:
                st.metric("Disease Data Points Monitored", len(df_disease))
            
            st.markdown("---")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("#### 🪵 Top Adverse Reactions Frequency")
                if not df_fda.empty:
                    reaction_counts = df_fda['adverse_reaction'].value_counts().head(10)
                    st.bar_chart(reaction_counts)
                else:
                    st.info("No FDA Data available yet.")
                    
            with col2:
                st.markdown("#### 📈 Daily New Cases Trend (Disease.sh)")
                if not df_disease.empty:
                    df_disease['date'] = pd.to_datetime(df_disease['date'])
                    df_disease = df_disease.sort_values('date')
                    st.line_chart(df_disease.set_index('date')['daily_new_cases'])
                else:
                    st.info("No Disease Metrics synced yet.")
                    
        except Exception as e:
            st.error(f"Error rendering analytics: {e}")

    with tab2:
        st.subheader("🔮 Predictive ML Inference")
        
        if model_metadata is None:
            st.warning("⚠️ Model not found! GitHub Actions will train it automatically on next run.")
        else:
            model = model_metadata['model']
            feature_columns = model_metadata['feature_columns']
            
            st.markdown("Enter patient details to get **Real-Time Risk Score** from XGBoost model.")
            
            with st.form("patient_inference_form"):
                col_f1, col_f2 = st.columns(2)
                
                with col_f1:
                    age = st.slider("Patient Age", min_value=1, max_value=100, value=40)
                    gender = st.selectbox("Patient Gender", ["Male", "Female", "Unknown"])
                
                with col_f2:
                    trained_reactions = [
                        col.replace("adverse_reaction_cleaned_", "")
                        for col in feature_columns
                        if col.startswith("adverse_reaction_cleaned_")
                    ] or ["OTHER"]
                    reaction = st.selectbox("Observed Symptom", sorted(trained_reactions + ["OTHER"]))
                
                submit_btn = st.form_submit_button("🔍 Compute Risk Score")
                
                if submit_btn:
                    input_df = pd.DataFrame(0, index=[0], columns=feature_columns)
                    input_df['patient_age'] = age
                    
                    gender_col = f"patient_gender_{gender}"
                    reaction_col = f"adverse_reaction_cleaned_{reaction}"
                    
                    if gender_col in input_df.columns:
                        input_df[gender_col] = 1
                    if reaction_col in input_df.columns:
                        input_df[reaction_col] = 1
                    
                    prediction = model.predict(input_df)[0]
                    probabilities = model.predict_proba(input_df)[0]
                    
                    st.markdown("### 📊 Risk Classification Result")
                    if prediction == 1:
                        st.error("⚠️ **HIGH RISK — Serious Adverse Event Predicted**")
                        st.progress(float(probabilities[1]))
                        st.write(f"High Risk: **{probabilities[1]*100:.2f}%** | Low Risk: {probabilities[0]*100:.2f}%")
                    else:
                        st.success("✅ **LOW RISK — Mild Event Predicted**")
                        st.progress(float(probabilities[0]))
                        st.write(f"Low Risk: **{probabilities[0]*100:.2f}%** | High Risk: {probabilities[1]*100:.2f}%")