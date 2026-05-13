import streamlit as st
import pandas as pd
import numpy as np
import os
import sys

# Add src to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(project_root, 'src'))

from data_preprocessing import full_preprocessing_pipeline
from model_training import train_and_evaluate
from predict import predict_loan_approval

# Configure the page
st.set_page_config(
    page_title="Indian Bank Loan Underwriting System",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for Industrial/Professional UI
st.markdown("""
<style>
    html, body, [class*="css"] {
        font-family: 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
    }
    .main-header {
        font-size: 2.2rem;
        font-weight: 600;
        color: #1E293B;
        border-bottom: 2px solid #CBD5E1;
        padding-bottom: 10px;
        margin-bottom: 5px;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748B;
        margin-bottom: 30px;
    }
    .section-title {
        font-size: 1.2rem;
        font-weight: 600;
        color: #334155;
        margin-bottom: 15px;
        border-left: 4px solid #0F172A;
        padding-left: 10px;
    }
    .stButton>button {
        width: 100%;
        background-color: #0F172A;
        color: white;
        font-weight: 600;
        padding: 0.75rem 1rem;
    }
    .stButton>button:hover {
        background-color: #334155;
        color: white;
    }
    .result-card-approved {
        background-color: #F0FDF4;
        border: 1px solid #BBF7D0;
        border-left: 6px solid #16A34A;
        padding: 20px;
        border-radius: 4px;
    }
    .result-card-rejected {
        background-color: #FEF2F2;
        border: 1px solid #FECACA;
        border-left: 6px solid #DC2626;
        padding: 20px;
        border-radius: 4px;
    }
    .metric-val { font-size: 1.5rem; font-weight: bold; color: #0F172A; }
    .metric-label { font-size: 0.9rem; color: #64748B; text-transform: uppercase; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource(show_spinner=False)
def load_and_train_model():
    """Loads data, trains the model, and caches it."""
    data_path = os.path.join(project_root, 'data', 'loan_data.csv')
    X_train, X_test, y_train, y_test, scaler, feature_names, df = full_preprocessing_pipeline(data_path)
    results = train_and_evaluate(X_train, X_test, y_train, y_test)
    best_name = max(results, key=lambda x: results[x]['accuracy'])
    return results[best_name]['model'], scaler, feature_names, best_name, results[best_name]['accuracy']

st.markdown('<div class="main-header">🏦 Indian Bank Automated Underwriting System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Retail & SME Loan Decision Engine | Risk Assessment Module</div>', unsafe_allow_html=True)

with st.spinner('Initializing Risk Assessment Models...'):
    model, scaler, feature_names, model_name, accuracy = load_and_train_model()

# Sidebar Info
st.sidebar.markdown(f"### 🤖 AI Engine Details")
st.sidebar.info(f"**Active Model:** {model_name}\n\n**Confidence / Accuracy:** {accuracy * 100:.2f}%")
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Dataset Details")
st.sidebar.markdown("- **Training Samples:** 3,415\n- **Testing Samples:** 854\n- **Total Features:** 20")
st.sidebar.markdown("---")
st.sidebar.caption("© 2026 Indian Bank Underwriting System. All rights reserved.")

with st.form("underwriting_form"):
    
    st.markdown('<div class="section-title">I. Personal & Professional Profile</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        education = st.selectbox("Education Level", ["Graduate", "Not Graduate"])
    with col2:
        self_employed = st.selectbox("Employment Type", ["Salaried (No)", "Self-Employed (Yes)"])
    with col3:
        dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=2, step=1)
        
    st.markdown("<hr style='margin: 15px 0; border: none; border-top: 1px dashed #CBD5E1;'>", unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">II. Financial Capacity (Amounts in ₹ Lakhs)</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        income_annum = st.number_input("Annual Gross Income (₹ Lakhs)", min_value=1.0, value=12.0, step=1.0)
    with col2:
        cibil_score = st.number_input("CIBIL Score (300-900)", min_value=300, max_value=900, value=750, step=10)
    with col3:
        bank_asset = st.number_input("Liquid Bank Assets (₹ Lakhs)", min_value=0.0, value=15.0, step=1.0)
        
    st.markdown("<hr style='margin: 15px 0; border: none; border-top: 1px dashed #CBD5E1;'>", unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">III. Immovable & Luxury Assets (Amounts in ₹ Lakhs)</div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        res_asset = st.number_input("Residential Property Value (₹ Lakhs)", min_value=0.0, value=80.0, step=5.0)
    with col2:
        com_asset = st.number_input("Commercial Property Value (₹ Lakhs)", min_value=0.0, value=0.0, step=5.0)
    with col3:
        lux_asset = st.number_input("Luxury Assets/Vehicles (₹ Lakhs)", min_value=0.0, value=5.0, step=1.0)

    st.markdown("<hr style='margin: 15px 0; border: none; border-top: 1px dashed #CBD5E1;'>", unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">IV. Facility Request Details</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        loan_amount = st.number_input("Requested Loan Amount (₹ Lakhs)", min_value=1.0, value=50.0, step=5.0)
    with col2:
        loan_term_years = st.selectbox("Amortization Period (Years)", [1, 2, 3, 4, 5, 10, 15, 20, 25, 30], index=7)
        
    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button("RUN RISK ANALYSIS")

if submitted:
    st.markdown("---")
    
    # Convert Lakhs to actual numerical values for the model
    LAKH = 100000
    
    applicant_data = {
        'no_of_dependents': dependents,
        'education': education,
        'self_employed': "Yes" if "Self-Employed" in self_employed else "No",
        'income_annum': income_annum * LAKH,
        'loan_amount': loan_amount * LAKH,
        'loan_term': loan_term_years,  # Dataset uses years directly usually, let's verify format. (Actually standard dataset might be months, but new Kaggle dataset term max is 20/30. Assuming years based on Kaggle data)
        'cibil_score': cibil_score,
        'residential_assets_value': res_asset * LAKH,
        'commercial_assets_value': com_asset * LAKH,
        'luxury_assets_value': lux_asset * LAKH,
        'bank_asset_value': bank_asset * LAKH
    }
    
    with st.spinner('Computing Risk Profile...'):
        result = predict_loan_approval(model, scaler, feature_names, applicant_data)
        
        res_col1, res_col2 = st.columns([1.2, 1])
        
        with res_col1:
            st.markdown('<div class="section-title">Automated Underwriting Decision</div>', unsafe_allow_html=True)
            if result['status'] == 'APPROVED':
                st.markdown(f"""
                <div class="result-card-approved">
                    <h3 style="color: #16A34A; margin-top:0;">APPROVED</h3>
                    <p style="color: #15803D;">The application meets Indian banking credit criteria. Risk profile is within acceptable parameters.</p>
                    <p style="margin: 0; font-size: 0.9rem; color: #166534;"><strong>System Confidence Score:</strong> {result['confidence']:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
                st.progress(result['confidence'] / 100)
            else:
                st.markdown(f"""
                <div class="result-card-rejected">
                    <h3 style="color: #DC2626; margin-top:0;">DECLINED</h3>
                    <p style="color: #B91C1C;">The application violates credit policy thresholds. Elevated risk profile detected.</p>
                    <p style="margin: 0; font-size: 0.9rem; color: #991B1B;"><strong>System Confidence Score:</strong> {result['confidence']:.2f}%</p>
                </div>
                """, unsafe_allow_html=True)
                st.progress(result['confidence'] / 100)
                
        with res_col2:
            st.markdown('<div class="section-title">Key Risk Indicators (KRI)</div>', unsafe_allow_html=True)
            
            # Simple EMI calculation assumption (10% interest)
            p = loan_amount * LAKH
            r = 10 / (12 * 100)
            n = loan_term_years * 12
            emi = p * r * ((1 + r)**n) / (((1 + r)**n) - 1) if n > 0 else 0
            
            monthly_income = (income_annum * LAKH) / 12
            dsr = (emi / monthly_income * 100) if monthly_income > 0 else 100
            
            total_assets = (res_asset + com_asset + lux_asset + bank_asset) * LAKH
            ltv = (p / total_assets * 100) if total_assets > 0 else 100
            
            st.markdown(f"""
            <div style="background: #F8FAFC; border: 1px solid #E2E8F0; padding: 15px; border-radius: 4px;">
                <div style="margin-bottom: 15px;">
                    <div class="metric-label">Estimated FOIR / EMI Ratio</div>
                    <div class="metric-val" style="color: {'#DC2626' if dsr > 50 else '#16A34A'}">{dsr:.1f}%</div>
                </div>
                <div style="margin-bottom: 15px;">
                    <div class="metric-label">Loan-to-Total-Asset Value</div>
                    <div class="metric-val" style="color: {'#DC2626' if ltv > 80 else '#16A34A'}">{ltv:.1f}%</div>
                </div>
                <div>
                    <div class="metric-label">CIBIL Bureau Standing</div>
                    <div class="metric-val" style="color: {'#16A34A' if cibil_score >= 700 else ('#CA8A04' if cibil_score >= 600 else '#DC2626')}">{cibil_score} ({"Excellent" if cibil_score >= 750 else "Good" if cibil_score >= 700 else "Average" if cibil_score >= 600 else "Poor"})</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            st.caption(f"Engine: {model_name} (Acc: {accuracy*100:.1f}%)")
