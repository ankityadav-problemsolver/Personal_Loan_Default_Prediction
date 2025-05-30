import streamlit as st
import joblib
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc
import time
import os
from time import strftime
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
from scipy.stats import gaussian_kde
from model_metrics import (
    engineer_features,
    align_features,
    plot_roc,
    plot_precision_recall,
    plot_confusion_matrix,
    generate_classification_report
)

# Configuration
st.set_page_config(
    page_title="LoanRisk AI | Smart Credit Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Dark Color Scheme
COLOR_SCHEME = {
    'primary': '#636EFA',  # Vibrant blue
    'secondary': '#EF553B',  # Vibrant red
    'background': '#121721',  # Dark blue-gray
    'success': '#00CC96',  # Teal
    'danger': '#FF6692',  # Pink
    'warning': '#FECB52',  # Yellow
    'info': '#AB63FA',  # Purple
    'text': '#E2E2E2',  # Light gray
    'light': '#2A3A4E',  # Lighter dark
    'white': '#FFFFFF',
    'header': '#A6B7D4',  # Light blue-gray
    'dark': '#0D1117',  # Dark background
    'card': '#1E293B',  # Card background
    'highlight': '#FFA500'  # Orange for highlighting
}



import streamlit as st
from streamlit_lottie import st_lottie
import requests

# Load JSON animation from Lottie URL
def load_lottieurl(url: str):
    try:
        r = requests.get(url)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"Error loading Lottie: {e}")
    return None

# 🎨 Lottie animation URLs (verified & working)
sidebar_anim = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_t9gkkhz4.json")
# main_anim = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_cyuxhbnc.json")
main_anim = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_mjlh3hcy.json") 
chart_anim = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_UJNc2t.json")
footer_anim = load_lottieurl("https://assets1.lottiefiles.com/packages/lf20_jtbfg2nb.json")
skills_anim = load_lottieurl("https://assets3.lottiefiles.com/private_files/lf30_5ttqpi.json")

# 🎯 Sidebar with animation
with st.sidebar:
    st.markdown("### 🤖 Smart AI Dashboard")
    if sidebar_anim:
        st_lottie(sidebar_anim, height=180)
    st.markdown("Welcome! Use the menu to explore predictions and risk factors.")

# 🎯 Header
st.markdown("## 📊 Loan Risk Prediction AI Dashboard")
st.markdown("""
### 📌 Overview  
This tool helps banks evaluate loan applications using AI-driven analytics.
""")

# 🎯 Main AI animation
if main_anim:
    st_lottie(main_anim, height=260)




     # Analytics mascot
          # Data analysis
chart_anim = load_lottieurl("https://assets2.lottiefiles.com/packages/lf20_49rdyysj.json")          # Graphs interaction        # Thank you # Project showreel
loan_risk_anim = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_kyu7xb1v.json")




# Premium Custom CSS with Dark Theme
st.markdown(f"""
<style>
    /* Base Styles */
    .main {{
        background: {COLOR_SCHEME['background']};
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        color: {COLOR_SCHEME['text']};
    }}
    
    /* Text and Headings */
    h1 {{
        color: {COLOR_SCHEME['primary']} !important;
        font-weight: 700;
        margin-bottom: 0.5rem;
        border-bottom: 2px solid {COLOR_SCHEME['primary']};
        padding-bottom: 10px;
    }}
    h2 {{
        color: {COLOR_SCHEME['secondary']} !important;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }}
    h3 {{
        color: {COLOR_SCHEME['header']} !important;
        font-weight: 600;
    }}
    .stMarkdown p {{
        color: {COLOR_SCHEME['text']} !important;
        line-height: 1.6;
    }}
    
    /* Form Elements */
    .stNumberInput input, .stSlider input {{
        border: 1px solid {COLOR_SCHEME['primary']} !important;
    }}
    .stSlider .st-ae {{
        background: {COLOR_SCHEME['primary']} !important;
    }}
    
    /* Cards and Containers */
    .metric-card {{
        background: {COLOR_SCHEME['card']};
        border-radius: 12px;
        padding: 20px;
        margin: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        border-left: 4px solid {COLOR_SCHEME['primary']};
        transition: all 0.3s ease;
    }}
    .metric-card:hover {{
        transform: translateY(-3px);
        box-shadow: 0 8px 16px rgba(0,0,0,0.4);
    }}
    .graph-container {{
        background: {COLOR_SCHEME['card']};
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 25px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }}
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 10px;
    }}
    .stTabs [data-baseweb="tab"] {{
        background: {COLOR_SCHEME['light']};
        border-radius: 8px 8px 0 0 !important;
        padding: 10px 20px;
        transition: all 0.3s ease;
        color: {COLOR_SCHEME['text']};
    }}
    .stTabs [aria-selected="true"] {{
        background: {COLOR_SCHEME['primary']} !important;
        color: {COLOR_SCHEME['white']} !important;
    }}
    
    /* Buttons */
    .stButton>button {{
        background-color: {COLOR_SCHEME['primary']};
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        border: none;
        font-weight: 500;
        transition: all 0.3s ease;
    }}
    .stButton>button:hover {{
        background-color: {COLOR_SCHEME['secondary']};
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }}
    
    /* Form Elements */
    .stNumberInput, .stSlider, .stSelectbox {{
        margin-bottom: 20px;
    }}
    .stTextInput>div>div>input {{
        color: {COLOR_SCHEME['text']};
        background-color: {COLOR_SCHEME['card']};
    }}
    
    /* Footer */
    .footer {{
        background: linear-gradient(135deg, {COLOR_SCHEME['primary']}, {COLOR_SCHEME['secondary']});
        color: white;
        padding: 20px;
        text-align: center;
        margin-top: 40px;
        border-radius: 12px;
    }}
    .footer p {{
        color: white !important;
        margin: 5px 0;
    }}
    
    /* Animations */
    @keyframes fadeIn {{
        from {{ opacity: 0; transform: translateY(20px); }}
        to {{ opacity: 1; transform: translateY(0); }}
    }}
    .animated-card {{
        animation: fadeIn 0.6s ease-out;
    }}
    
    /* Highlight Box */
    .highlight-box {{
        padding: 20px;
        border-radius: 10px;
        color:black !important;
        margin: 20px 0;

        border-left: 5px solid {COLOR_SCHEME['highlight']};
    }}
    
    /* Plotly chart background */
    .js-plotly-plot .plotly, .js-plotly-plot .plotly div {{
        background: {COLOR_SCHEME['card']} !important;
    }}
</style>
""", unsafe_allow_html=True)

# Load resources with error handling and caching optimizations
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        model_path = os.path.join('best_model', 'best_model_Random_Forest.pkl')
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

@st.cache_data(show_spinner=False)
def load_data():
    try:
        data_path = os.path.join('notebooks', 'cleaned_loan_predictions.csv')
        df = pd.read_csv(data_path)
        
        # Convert categorical columns to numeric for ROC curve
        if 'REASON' in df.columns:
            df['REASON'] = df['REASON'].map({'HomeImp': 0, 'DebtCon': 1})
        if 'JOB' in df.columns:
            df['JOB'] = pd.factorize(df['JOB'])[0]
            
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return pd.DataFrame()

# Load resources with progress indicators
with st.spinner('Loading model and data...'):
    model = load_model()
    df = load_data()



# import os
# import json
# import joblib
# import pandas as pd
# import streamlit as st
# from datetime import datetime

# # Configuration
# MODEL_DIR = "best_model"
# DATA_PATH = os.path.join('notebooks', 'cleaned_loan_predictions.csv')

# def get_latest_model():
#     """Find the latest model and its metadata"""
#     try:
#         # Get all metadata files
#         meta_files = [f for f in os.listdir(MODEL_DIR) 
#                     if f.endswith('_metadata.json')]
        
#         if not meta_files:
#             st.error("No model metadata files found")
#             return None, None, None
            
#         # Sort by creation date (newest first)
#         meta_files.sort(reverse=True)
#         latest_meta = meta_files[0]
        
#         # Load metadata
#         meta_path = os.path.join(MODEL_DIR, latest_meta)
#         with open(meta_path) as f:
#             metadata = json.load(f)
        
#         # Verify model file exists - fix path issue
#         model_path = metadata.get('model_path')
#         if model_path and not os.path.isabs(model_path):
#             model_path = os.path.join(MODEL_DIR, os.path.basename(model_path))
            
#         if not model_path or not os.path.exists(model_path):
#             st.error(f"Model file not found: {model_path}")
#             return None, None, None
            
#         return model_path, meta_path, metadata
        
#     except Exception as e:
#         st.error(f"Model discovery error: {str(e)}")
#         return None, None, None

# def get_data_status():
#     """Check data file status for freshness"""
#     try:
#         return os.path.getmtime(DATA_PATH) if os.path.exists(DATA_PATH) else None
#     except Exception as e:
#         st.error(f"Data status check failed: {str(e)}")
#         return None

# # Resource loading with smart caching
# @st.cache_resource(show_spinner=False, ttl=3600)
# def load_model(model_path):
#     """Load model with validation"""
#     try:
#         model = joblib.load(model_path)
#         # Moved toast notification outside cached function
#         return model
#     except Exception as e:
#         st.error(f"Model load failed: {str(e)}")
#         return None

# # Modified load_data to remove Streamlit elements from cached function
# @st.cache_data(show_spinner=False, ttl=300)
# def load_data(_data_status):
#     """Load and preprocess data without Streamlit elements"""
#     try:
#         df = pd.read_csv(DATA_PATH)
        
#         # Dynamic preprocessing
#         category_mappings = {
#             'REASON': {'HomeImp': 0, 'DebtCon': 1},
#             'JOB': lambda x: pd.factorize(x)[0]
#         }
        
#         for col, mapping in category_mappings.items():
#             if col in df.columns:
#                 df[col] = df[col].map(mapping) if isinstance(mapping, dict) else mapping(df[col])
        
#         return df
#     except Exception as e:
#         return pd.DataFrame()

# # Main loading process
# with st.spinner('Initializing application...'):
#     # Dynamic model loading
#     model_path, meta_path, metadata = get_latest_model()
#     model = load_model(model_path) if model_path else None
    
#     # Real-time data loading
#     data_status = get_data_status()
#     df = load_data(data_status) if data_status else pd.DataFrame()

#     # Show notifications after loading
#     if model_path:
#         st.toast(f"Model loaded: {os.path.basename(model_path)}", icon="🤖")
#     if not df.empty:
#         st.toast("Data successfully refreshed", icon="🔄")



# Custom CSS for cohesive visual design
st.markdown("""
    <style>
    :root {
        --primary-dark: #0A192F;    /* Deep Navy */
        --secondary-dark: #172A45;  /* Medium Navy */
        --accent-teal: #64FFDA;     /* Bright Teal */
        --text-light: #CCD6F6;      /* Soft White */
    }

    .dashboard-header {
        background: var(--primary-dark);
        border-left: 5px solid var(--accent-teal);
        padding: 2rem;
        border-radius: 8px;
        margin-bottom: 2rem;
    }

    .metric-card {
        background: var(--secondary-dark);
        border-radius: 10px;
        padding: 1.5rem;
        border: 1px solid rgba(100, 255, 218, 0.15);
        transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }

    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 15px rgba(100, 255, 218, 0.2);
    }

    .metric-value {
        font-size: 2.2rem;
        font-weight: 600;
        color: var(--accent-teal);
        margin: 0.5rem 0;
    }

    .info-badge {
        background: rgba(100, 255, 218, 0.1);
        color: var(--accent-teal);
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)

# # Dashboard Implementation
# if model and metadata:
#     st.markdown(f"""
#         <div class="dashboard-header">
#             <h1 style="color: var(--text-light); margin-bottom: 1rem;">🌌 Model Intelligence Dashboard</h1>
#             <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 2rem; color: var(--text-light);">
#                 <div>
#                     <div class="info-badge">Model Name</div>
#                     <p style="font-size: 1.2rem; margin: 0.5rem 0;">{metadata['model_name']}</p>
#                 </div>
#                 <div>
#                     <div class="info-badge">Version</div>
#                     <p style="font-size: 1.2rem; margin: 0.5rem 0;">{metadata['version'].split('_')[0]}</p>
#                 </div>
#                 <div>
#                     <div class="info-badge">Created</div>
#                     <p style="font-size: 1.2rem; margin: 0.5rem 0;">
#                         {datetime.strptime(metadata['version'], "%Y%m%d_%H%M%S").strftime("%Y-%m-%d %H:%M")}
#                     </p>
#                 </div>
#             </div>
#         </div>
#     """, unsafe_allow_html=True)

# # Performance Metrics Grid
# metrics = metadata.get('performance_metrics', {})
# st.markdown("""
#     <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem; margin-bottom: 2rem;">
# """, unsafe_allow_html=True)

# metric_config = [
#     ("🎯 Accuracy", metrics.get('Accuracy', 0), "Percentage of correct predictions"),
#     ("🎯 Precision", metrics.get('Precision', 0), "True positive rate"),
#     ("🎯 Recall", metrics.get('Recall', 0), "Positive class coverage"),
#     ("🎯 F1 Score", metrics.get('F1 Score', 0), "Harmonic mean balance"),
#     ("📈 AUC-ROC", metrics.get('AUC', 0), "Classification strength score")
# ]

# for title, value, desc in metric_config:
#     # Format value based on metric type first
#     if title == "📈 AUC-ROC":
#         formatted_value = f"{float(value):.2f}"  # Ensure float conversion
#     else:
#         formatted_value = f"{float(value):.2%}"  # Ensure float conversion
    
#     st.markdown(f"""
#         <div class="metric-card">
#             <h4 style="color: var(--accent); margin: 0 0 1rem 0;">{title}</h4>
#             <div class="metric-value">{formatted_value}</div>
#             <p style="color: var(--text); opacity: 0.8; font-size: 0.9rem; margin: 0;">{desc}</p>
#         </div>
#     """, unsafe_allow_html=True)

# st.markdown("</div>", unsafe_allow_html=True)


# Sidebar Configuration
with st.sidebar:
    st.markdown(f"""
    <div style="text-align:center; margin-bottom:30px;">
        <h3 style="color:{COLOR_SCHEME['primary']};">Dashboard Filters</h3>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize filters with safe defaults
    job_options = df['JOB'].unique() if 'JOB' in df.columns else []
    reason_options = ['HomeImp', 'DebtCon'] if 'REASON' in df.columns else []
    
    job_filter = st.multiselect("Select Occupation", options=job_options)
    reason_filter = st.multiselect("Select Loan Purpose", options=reason_options)

# Apply filters to data safely
filtered_df = df.copy()
if job_filter and 'JOB' in df.columns:
    filtered_df = filtered_df[filtered_df['JOB'].isin(job_filter)]
if reason_filter and 'REASON' in df.columns:
    filtered_df = filtered_df[filtered_df['REASON'].isin(reason_filter)]

# Create tabs with custom styling
tab1, tab2 = st.tabs(["🔍 Risk Assessment", "📊 Data Insights"])

# Risk Assessment Tab
with tab1:
    st.markdown(f"""
    <div style="margin-bottom:30px;">
        <h1 style="color:{COLOR_SCHEME['header']};">Loan Risk Assessment</h1>
        <p style="font-size:1.1rem; color:{COLOR_SCHEME['text']};">Evaluate borrower risk using our AI-powered scoring model</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Prediction Form
    # Prediction Form
    with st.form("loan_appraisal"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"#### <span style='color:{COLOR_SCHEME['primary']}'>🏦 Borrower Information</span>", unsafe_allow_html=True)

            loan = st.number_input("💰 Loan Amount (USD) *", 1000, 1000000, 5000,
                                help="Total loan amount requested by the borrower.")
            value = st.number_input("🏠 Property Value (USD) *", 10000, 2000000, 250000,
                                    help="Market value of the property used as collateral.")
            mortdue = st.number_input("📉 Existing Mortgage (USD)", 0, 1000000, 100000,
                                    help="Outstanding balance on any existing mortgages.")
            debtinc = st.slider("⚖️ Debt-to-Income Ratio (%) *", 0.0, 100.0, 35.0, 0.5,
                                help="A high ratio indicates higher financial risk.")

        with col2:
            st.markdown(f"#### <span style='color:{COLOR_SCHEME['secondary']}'>📋 Credit History</span>", unsafe_allow_html=True)

            yoj = st.slider("🧑‍💼 Years at Current Job", 0, 40, 5,
                            help="Stability of employment, relevant to creditworthiness.")
            derog = st.number_input("❗ Derogatory Reports *", 0, 10, 0,
                                    help="Any serious negative credit reports. Affects risk significantly.")
            delinq = st.number_input("⚠️ Delinquent Accounts", 0, 10, 0,
                                    help="Number of accounts overdue or in collections.")
            clage = st.number_input("📆 Oldest Credit Line (Months)", 0, 500, 120,
                                    help="Age of the oldest credit line (longer is usually better).")
            ninq = st.number_input("🔍 Recent Credit Inquiries", 0, 10, 0,
                                help="Too many inquiries may lower credit scores.")

        st.markdown("#### 📌 Additional Information")
        reason = st.selectbox("🎯 Loan Purpose (REASON) *", ['DebtCon', 'HomeImp'],
                            help="Reason for loan: 'DebtCon' = debt consolidation, 'HomeImp' = home improvement.")
        job = st.selectbox("🧑 Employment Type (JOB) *", ['Other', 'Mgr', 'Office', 'Sales', 'ProfExe', 'Self'],
                        help="Job category affects financial stability prediction.")

        submitted = st.form_submit_button("🔍 Assess Risk", type="primary")

    # Optional Tip Box for User
    st.markdown(f"""
    <div style="padding:1rem; background-color:{COLOR_SCHEME['card']}; border-left: 5px solid {COLOR_SCHEME['primary']}; color:{COLOR_SCHEME['text']};">
    <b>🧠 Tip:</b> Fields marked with <span style="color:red">*</span> are critical to your credit risk assessment. 
    Focus especially on <b>Loan Amount</b>, <b>Debt-to-Income Ratio</b>, and <b>Derogatory Reports</b> as they significantly influence the model's classification.
    </div>
    """, unsafe_allow_html=True)

    if submitted and model is not None:
        try:
            # Input validation
            if value <= 0:
                st.warning("Property value must be positive")
            if loan <= 0:
                st.warning("Loan amount must be positive")
            if mortdue < 0:
                st.warning("Mortgage balance cannot be negative")
            if debtinc < 0 or debtinc > 100:
                st.warning("Debt-to-income ratio must be between 0-100%")

            # Prepare input data
            input_data = {
                'LOAN': loan,
                'VALUE': value,
                'MORTDUE': mortdue,
                'DEBTINC': debtinc,
                'YOJ': yoj,
                'DEROG': derog,
                'DELINQ': delinq,
                'CLAGE': clage,
                'NINQ': ninq,
                'CLNO': 20,
                'REASON': reason,
                'JOB': job
            }

            input_df = pd.DataFrame([input_data])

            # Feature Engineering
            input_df['LOAN_TO_VALUE'] = input_df['LOAN'] / input_df['VALUE']
            input_df['LOAN_TO_MORTDUE'] = input_df['LOAN'] / (input_df['MORTDUE'] + 1)
            input_df['DEROG_DELINQ_SUM'] = input_df['DEROG'] + input_df['DELINQ']
            input_df['CLAGE_PER_CLNO'] = input_df['CLAGE'] / (input_df['CLNO'] + 1)

            # EMI calculation
            monthly_rate = 0.07 / 12
            loan_term = 60
            input_df['EMI'] = (input_df['LOAN'] * monthly_rate * (1 + monthly_rate)**loan_term) / \
                            ((1 + monthly_rate)**loan_term - 1)

            # Binning
            input_df['YOJ_BINNED'] = pd.cut(input_df['YOJ'], 
                                            bins=[-1, 2, 5, 10, 20, 40], 
                                            labels=['0-2', '2-5', '5-10', '10-20', '20+'])
            input_df['CLAGE_BINNED'] = pd.cut(input_df['CLAGE'], 
                                            bins=[-1, 60, 120, 180, 240, 500], 
                                            labels=['0-5', '5-10', '10-15', '15-20', '20+'])

            # Encoding
            input_df['REASON'] = input_df['REASON'].map({'HomeImp': 0, 'DebtCon': 1})
            input_df['JOB'] = pd.factorize(input_df['JOB'])[0]
            input_df['YOJ_BINNED'] = pd.factorize(input_df['YOJ_BINNED'])[0]
            input_df['CLAGE_BINNED'] = pd.factorize(input_df['CLAGE_BINNED'])[0]

            # Strict feature alignment with model
            if hasattr(model, 'feature_names_in_'):
                # Remove any extra columns not in model's features
                input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)
            else:
                # Fallback for models without feature names (legacy)
                st.warning("⚠️ Model feature names not available - ensure manual alignment")
                input_df = input_df.iloc[:, :19]  # Keep only first 19 features if needed

            # Prediction
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]

            # Result Display
            if prediction == 1:
                st.markdown(f"""
                <div class="highlight-box">
                    <h2 style="color:{COLOR_SCHEME['danger']};">🚨 High Risk Alert</h2>
                    <p style="font-size:1.2rem;">Probability of Default: <strong>{probability:.1%}</strong></p>
                    <p>This application shows significant risk factors based on our analysis.</p>
                </div>
                """, unsafe_allow_html=True)
                
            else:
                st.markdown(f"""
                <div class="highlight-box">
                    <h2 style="color:{COLOR_SCHEME['success']};">✅ Low Risk</h2>
                    <p style="font-size:1.2rem;">Probability of Default: <strong>{probability:.1%}</strong></p>
                    <p>This application meets our credit standards.</p>
                </div>
                """, unsafe_allow_html=True)
            # SHAP Explanation
            

        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
        
    # # Model Insights Section
    # if model is not None:
    #     st.markdown("---")
    #     st.markdown("## Model Performance Analytics")
        
    #     # ROC Curve with proper data handling
    #     if 'BAD' in df.columns:
    #         try:
    #             with st.spinner('Calculating model performance metrics...'):
    #                 X = df.drop('BAD', axis=1)
    #                 y = df['BAD']
                    
    #                 # Align columns with model's expected features
    #                 if hasattr(model, 'feature_names_in_'):
    #                     available_features = [col for col in model.feature_names_in_ if col in X.columns]
    #                     X = X[available_features]
                        
    #                     # Add missing features with default values
    #                     missing_features = set(model.feature_names_in_) - set(X.columns)
    #                     for feature in missing_features:
    #                         X[feature] = 0
                        
    #                     # Reorder columns to match model expectations
    #                     X = X[model.feature_names_in_]
                    
    #                 probas = model.predict_proba(X)[:, 1]
    #                 fpr, tpr, thresholds = roc_curve(y, probas)
    #                 roc_auc = auc(fpr, tpr)
                    
    #                 with st.container():
    #                     st.markdown("#### Model Discrimination Ability (ROC Curve)")
    #                     fig = go.Figure()
    #                     fig.add_trace(go.Scatter(
    #                         x=fpr, y=tpr,
    #                         mode='lines',
    #                         line=dict(color=COLOR_SCHEME['primary'], width=3),
    #                         name=f'ROC Curve (AUC = {roc_auc:.2f})'
    #                     ))
    #                     fig.add_trace(go.Scatter(
    #                         x=[0, 1], y=[0, 1],
    #                         mode='lines',
    #                         line=dict(color=COLOR_SCHEME['danger'], dash='dash'),
    #                         name='Random Guessing'
    #                     ))
    #                     fig.update_layout(
    #                         xaxis_title='False Positive Rate',
    #                         yaxis_title='True Positive Rate',
    #                         height=500,
    #                         plot_bgcolor=COLOR_SCHEME['card'],
    #                         paper_bgcolor=COLOR_SCHEME['background'],
    #                         font=dict(color=COLOR_SCHEME['text']),
    #                         margin=dict(l=50, r=50, b=50, t=50),
    #                         legend=dict(
    #                             orientation="h",
    #                             yanchor="bottom",
    #                             y=1.02,
    #                             xanchor="right",
    #                             x=1
    #                         )
    #                     )
    #                     st.plotly_chart(fig, use_container_width=True)
    #                     st.markdown(f"""
    #                     <div style="color:{COLOR_SCHEME['text']};">
    #                         <p>The ROC curve shows the trade-off between true positive rate (sensitivity) and false positive rate (1-specificity). 
    #                         Our model achieves an AUC of <strong>{roc_auc:.2f}</strong>, indicating good discrimination ability. 
    #                         An AUC of 1.0 represents perfect prediction, while 0.5 represents random guessing.</p>
    #                     </div>
    #                     """, unsafe_allow_html=True)
    #         except Exception as e:
    #             st.error(f"Error generating ROC curve: {str(e)}")
    
    



    # Assuming COLOR_SCHEME and df, model are already defined
    if model is not None:
        st.markdown("---")
        st.markdown("## Model Performance Analytics")

        if 'BAD' in df.columns:
            try:
                with st.spinner('Calculating model performance metrics...'):
                    # --- Feature Engineering ---
                    input_df = engineer_features(df)

                    # --- Prepare Input Data ---
                    X = input_df.drop('BAD', axis=1)
                    y = input_df['BAD']

                    # --- Align with model's expected input ---
                    X = align_features(X, model)

                    # --- Predict Probabilities and Labels ---
                    y_proba = model.predict_proba(X)[:, 1]
                    y_pred = model.predict(X)

                    # --- ROC Curve ---
                    roc_fig, auc_val = plot_roc(y, y_proba, COLOR_SCHEME)
                    st.markdown("### ROC Curve")
                    st.plotly_chart(roc_fig, use_container_width=True)
                    st.markdown(f"""
                        <div style="color:{COLOR_SCHEME['text']};">
                            <p>The ROC curve shows the trade-off between true positive rate (sensitivity) and false positive rate (1-specificity). 
                            Our model achieves an AUC of <strong>{auc_val:.2f}</strong>.</p>
                        </div>
                    """, unsafe_allow_html=True)

                    # --- Precision-Recall Curve ---
                    st.markdown("### Precision-Recall Curve")
                    pr_fig = plot_precision_recall(y, y_proba, COLOR_SCHEME)
                    st.plotly_chart(pr_fig, use_container_width=True)

                    # --- Confusion Matrix ---
                    st.markdown("### Confusion Matrix")
                    cm_fig = plot_confusion_matrix(y, y_pred, COLOR_SCHEME)
                    st.plotly_chart(cm_fig, use_container_width=True)

                    # --- Classification Report ---
                    st.markdown("### Classification Report")
                    report = generate_classification_report(y, y_pred)
                    st.text(report)

            except Exception as e:
                st.error(f"Error generating performance metrics: {e}")
            
            
            # Feature Importance with proper handling
            # Feature mapping dictionary - replace with your actual feature names
        FEATURE_MAPPING = {
            'Feature 0': 'Loan Amount',
            'Feature 1': 'Credit Score',
            'Feature 2': 'Debt-to-Income Ratio',
            'Feature 4': 'Employment Length',
            'Feature 5': 'Annual Income',
            'Feature 8': 'Loan Purpose',
            'Feature 9': 'Years at Current Address',
            'Feature 10': 'Number of Open Accounts',
            'Feature 11': 'Number of Credit Problems',
            'Feature 12': 'Current Credit Balance',
            'Feature 13': 'Maximum Open Credit',
            'Feature 14': 'Bankruptcies',
            'Feature 15': 'Tax Liens',
            'Feature 17': 'Number of Late Payments (30-59 days)',
            'Feature 18': 'Number of Late Payments (60-89 days)'
        }

        # Feature Importance with proper names
        if hasattr(model, 'feature_importances_'):
            try:
                with st.container():
                    st.markdown("#### Key Risk Drivers (Feature Importance)")
                    
                    # Get features and importances
                    if hasattr(model, 'feature_names_in_'):
                        features = model.feature_names_in_
                        importances = model.feature_importances_
                    else:
                        features = [f"Feature {i}" for i in range(len(model.feature_importances_))]
                        importances = model.feature_importances_
                    
                    # Map feature numbers to proper names
                    named_features = [FEATURE_MAPPING.get(f, f) for f in features]
                    
                    # Create DataFrame and sort
                    feat_imp = pd.DataFrame({
                        'Feature': named_features,
                        'Importance': importances,
                        'Original_Feature': features  # Keep original for reference
                    }).sort_values('Importance', ascending=True)
                    
                    # Create horizontal bar chart
                    fig = px.bar(
                        feat_imp.tail(15),  # Show top 15 features
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        color='Importance',
                        color_continuous_scale='Viridis',
                        title='',
                        hover_data=['Original_Feature']  # Show original feature number in tooltip
                    )
                    
                    fig.update_layout(
                        height=500,
                        yaxis={'categoryorder':'total ascending'},
                        margin=dict(l=100, r=50, b=50, t=50),
                        plot_bgcolor=COLOR_SCHEME['card'],
                        paper_bgcolor=COLOR_SCHEME['background'],
                        font=dict(color=COLOR_SCHEME['text']),
                        coloraxis_colorbar=dict(
                            title='Importance',
                            tickfont=dict(color=COLOR_SCHEME['text']))
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.markdown(f"""
                    <div style="color:{COLOR_SCHEME['text']};">
                        <p>This chart shows which factors most influence the model's risk assessment. 
                        Higher values indicate features that contribute more to predicting loan defaults. 
                        Understanding these drivers helps in making informed lending decisions and explaining model behavior.</p>
                        <p><small>Hover over bars to see the original feature numbers</small></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"Error generating feature importance: {str(e)}")

# Data Analytics Tab
with tab2:
    st.markdown(f"""
    <div style="margin-bottom:30px;">
        <h1 style="color:{COLOR_SCHEME['header']};">Loan Portfolio Insights</h1>
        <p style="font-size:1.1rem; color:{COLOR_SCHEME['text']};">Interactive visualizations of our loan portfolio characteristics</p>
    </div>
    """, unsafe_allow_html=True)
    
    if not filtered_df.empty:
        # KPI Cards
        st.markdown("### Portfolio Overview")
        cols = st.columns(4)
        with cols[0]:
            st.markdown(f"""
            <div class="metric-card animated-card">
                <h3>Total Loans</h3>
                <h2>{len(filtered_df):,}</h2>
            </div>
            """, unsafe_allow_html=True)
        with cols[1]:
            default_rate = filtered_df['BAD'].mean() if 'BAD' in filtered_df.columns else 0
            st.markdown(f"""
            <div class="metric-card animated-card">
                <h3>Default Rate</h3>
                <h2>{default_rate:.1%}</h2>
            </div>
            """, unsafe_allow_html=True)
        with cols[2]:
            avg_loan = filtered_df['LOAN'].mean() if 'LOAN' in filtered_df.columns else 0
            st.markdown(f"""
            <div class="metric-card animated-card">
                <h3>Avg Loan Amount</h3>
                <h2>${avg_loan:,.0f}</h2>
            </div>
            """, unsafe_allow_html=True)
        with cols[3]:
            avg_debtinc = filtered_df['DEBTINC'].mean() if 'DEBTINC' in filtered_df.columns else 0
            st.markdown(f"""
            <div class="metric-card animated-card">
                <h3>Avg Debt-to-Income</h3>
                <h2>{avg_debtinc:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
         # Pie Charts Section
        st.markdown("### Portfolio Composition")
        pie_col1, pie_col2 = st.columns(2)
        
        with pie_col1:
            if 'REASON' in filtered_df.columns:
                reason_counts = filtered_df['REASON'].value_counts()
                fig = px.pie(
                    reason_counts,
                    values=reason_counts.values,
                    names=reason_counts.index.map({0: 'Home Improvement', 1: 'Debt Consolidation'}),
                    title='Loan Purpose Distribution',
                    color_discrete_sequence=[COLOR_SCHEME['primary'], COLOR_SCHEME['secondary']]
                )
                fig.update_layout(
                    plot_bgcolor=COLOR_SCHEME['card'],
                    paper_bgcolor=COLOR_SCHEME['background'],
                    font=dict(color=COLOR_SCHEME['text']),
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with pie_col2:
            if 'BAD' in filtered_df.columns:
                status_counts = filtered_df['BAD'].value_counts()
                fig = px.pie(
                    status_counts,
                    values=status_counts.values,
                    names=status_counts.index.map({0: 'Good Loans', 1: 'Bad Loans'}),
                    title='Loan Status Distribution',
                    color_discrete_sequence=[COLOR_SCHEME['success'], COLOR_SCHEME['danger']]
                )
                fig.update_layout(
                    plot_bgcolor=COLOR_SCHEME['card'],
                    paper_bgcolor=COLOR_SCHEME['background'],
                    font=dict(color=COLOR_SCHEME['text']),
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
        
        # Distribution Plots
        st.markdown("### Feature Distributions")
        dist_col1, dist_col2 = st.columns(2)
        
        with dist_col1:
            if 'LOAN' in filtered_df.columns:


                # Extract loan data for KDE
                loan_data = filtered_df['LOAN'].dropna()

                # Create the Histogram
                hist = go.Histogram(
                    x=loan_data,
                    nbinsx=30,
                    marker_color=COLOR_SCHEME['primary'],
                    opacity=0.6,
                    name='Loan Amount Distribution'
                )

                # Generate the KDE line
                kde = gaussian_kde(loan_data)
                x_range = np.linspace(loan_data.min(), loan_data.max(), 500)
                kde_y = kde(x_range) * len(loan_data) * (loan_data.max() - loan_data.min()) / 30  # Normalize to match histogram scale

                # KDE Line plot
                kde_line = go.Scatter(
                    x=x_range,
                    y=kde_y,
                    mode='lines',
                    line=dict(color=COLOR_SCHEME['secondary'], width=2),
                    name='KDE Curve'
                )

                # Combine histogram and KDE line
                fig = go.Figure(data=[hist, kde_line])

                # Update Layout
                fig.update_layout(
                    plot_bgcolor=COLOR_SCHEME['card'],
                    paper_bgcolor=COLOR_SCHEME['background'],
                    font=dict(color=COLOR_SCHEME['text']),
                    height=400,
                    xaxis_title='Loan Amount (USD)',
                    yaxis_title='Count',
                    barmode='overlay'
                )

                # Display the Plotly chart in Streamlit
                st.plotly_chart(fig, use_container_width=True)

                # Display a description
                st.markdown(f"""
                <div style="color:{COLOR_SCHEME['text']};">
                    <p>This chart shows the distribution of loan amounts with a smooth Kernel Density Estimation (KDE) line 
                    representing the probability density. The KDE curve gives a smoother view of the distribution compared to the histogram.
                    Most loans cluster around ${filtered_df['LOAN'].median():,.0f}, with some larger outliers. 
                    Understanding this distribution aids in better product pricing and risk analysis.</p>
                </div>
                """, unsafe_allow_html=True)

        
        with dist_col2:
            if 'DEBTINC' in filtered_df.columns:
                fig = px.box(
                    filtered_df,
                    y='DEBTINC',
                    color_discrete_sequence=[COLOR_SCHEME['secondary']],
                    title='Debt-to-Income Ratio'
                )
                fig.update_layout(
                    plot_bgcolor=COLOR_SCHEME['card'],
                    paper_bgcolor=COLOR_SCHEME['background'],
                    font=dict(color=COLOR_SCHEME['text']),
                    height=400,
                    showlegend=False,
                    yaxis_title='Debt-to-Income Ratio (%)'
                )
                st.plotly_chart(fig, use_container_width=True)
                st.markdown(f"""
                <div style="color:{COLOR_SCHEME['text']};">
                    <p>Box plot showing the distribution of debt-to-income ratios. The median DTI is {filtered_df['DEBTINC'].median():.1f}%, 
                    with upper outliers indicating potentially risky borrowers.</p>
                </div>
                """, unsafe_allow_html=True)
        
        # 3D Visualizations
        st.markdown("### Multivariate Analysis")
        
        # 3D Scatter Plot
        if all(col in filtered_df.columns for col in ['LOAN', 'VALUE', 'DEBTINC', 'BAD']):
            st.markdown("#### Loan Amount vs Property Value vs DTI")
            sample_df = filtered_df.sample(min(1000, len(filtered_df)))
            fig = px.scatter_3d(
                sample_df,
                x='LOAN',
                y='VALUE',
                z='DEBTINC',
                color='BAD',
                color_continuous_scale=[COLOR_SCHEME['success'], COLOR_SCHEME['danger']],
                opacity=0.7,
                hover_name='JOB' if 'JOB' in sample_df.columns else None
            )
            fig.update_layout(
                height=700,
                scene=dict(
                    xaxis_title='Loan Amount',
                    yaxis_title='Property Value',
                    zaxis_title='Debt-to-Income',
                    xaxis=dict(backgroundcolor=COLOR_SCHEME['card']),
                    yaxis=dict(backgroundcolor=COLOR_SCHEME['card']),
                    zaxis=dict(backgroundcolor=COLOR_SCHEME['card']),
                ),
                margin=dict(l=0, r=0, b=0, t=30),
                paper_bgcolor=COLOR_SCHEME['background']
            )
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"""
            <div style="color:{COLOR_SCHEME['text']};">
                <p>This 3D visualization shows the relationship between loan amount, property value, and debt-to-income ratio. 
                Red points indicate defaulted loans. Look for clusters of red points to identify high-risk combinations.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Surface Plot for Default Rates
        if all(col in filtered_df.columns for col in ['YOJ', 'CLAGE', 'BAD']):
            st.markdown("#### Default Rate Surface by Employment and Credit Age")
            pivot_data = filtered_df.pivot_table(
                index='YOJ', 
                columns='CLAGE', 
                values='BAD', 
                aggfunc='mean'
            ).fillna(0)
            
            fig = go.Figure(data=[
                go.Surface(
                    z=pivot_data.values,
                    x=pivot_data.columns,
                    y=pivot_data.index,
                    colorscale='Viridis',
                    colorbar=dict(title='Default Rate')
                )
            ])
            
            fig.update_layout(
                scene=dict(
                    xaxis_title='Credit Age (Months)',
                    yaxis_title='Years at Job',
                    zaxis_title='Default Rate',
                    xaxis=dict(backgroundcolor=COLOR_SCHEME['card']),
                    yaxis=dict(backgroundcolor=COLOR_SCHEME['card']),
                    zaxis=dict(backgroundcolor=COLOR_SCHEME['card']),
                ),
                height=700,
                margin=dict(l=0, r=0, b=0, t=30),
                paper_bgcolor=COLOR_SCHEME['background']
            )
            
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(f"""
            <div style="color:{COLOR_SCHEME['text']};">
                <p>Surface plot showing how default rates vary with employment duration and credit age. 
                Higher peaks indicate riskier combinations of these factors.</p>
            </div>
            """, unsafe_allow_html=True)
        
# Mid-animation
        st.markdown("---")
        st.markdown("### 📉 Correlation Insights")
        st_lottie(chart_anim, height=180)
        # Correlation Heatmap
        st.markdown("### Feature Correlations")

        # Select only numeric columns
        numeric_df = filtered_df.select_dtypes(include=np.number)

        if numeric_df.shape[1] > 1:
            # Compute correlation
            corr_matrix = numeric_df.corr()

            # Fill NaN correlations with 0 for display purposes
            corr_matrix_filled = corr_matrix.fillna(0)

            fig = px.imshow(
                corr_matrix_filled,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                color_continuous_scale='RdBu',
                zmin=-1,
                zmax=1,
                text_auto=".2f"
            )

            fig.update_layout(
                height=600,
                xaxis_showgrid=False,
                yaxis_showgrid=False,
                plot_bgcolor=COLOR_SCHEME['card'],
                paper_bgcolor=COLOR_SCHEME['background'],
                font=dict(color=COLOR_SCHEME['text'])
            )

            st.plotly_chart(fig, use_container_width=True)

            # Markdown summary
            st.markdown(f"""
            <div style="color:{COLOR_SCHEME['text']};">
                <p>This heatmap displays correlation between numeric features. NaN values are replaced with 0 for clarity. 
                Use this to detect patterns or multicollinearity among features.</p>
            </div>
            """, unsafe_allow_html=True)
                
        
        
        st.markdown("### Default Rate by Occupation")
        if 'JOB' in filtered_df.columns and 'BAD' in filtered_df.columns:
            job_defaults = filtered_df.groupby('JOB')['BAD'].mean().reset_index()
            fig = px.bar(job_defaults, x='JOB', y='BAD',
                        color='BAD', color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
        
        
        
         # New: Animated Surface Plot with Performance Optimization
        st.subheader("Risk Surface Analysis")
        if st.button("Start Optimized Animation", key="animate_button"):
            if all(col in filtered_df.columns for col in ['YOJ', 'CLAGE', 'BAD']):
                try:
                    # Pre-calculate the pivot table
                    pivot_data = filtered_df.pivot_table(
                        index='YOJ', 
                        columns='CLAGE', 
                        values='BAD', 
                        aggfunc='mean'
                    ).fillna(0)
                    
                    # Create figure once
                    fig = go.Figure()
                    
                    # Add surface trace
                    fig.add_trace(go.Surface(
                        z=pivot_data.values,
                        colorscale='Electric',
                        x=pivot_data.columns,
                        y=pivot_data.index
                    ))
                    
                    # Set initial layout
                    fig.update_layout(
                        scene=dict(
                            xaxis_title='Credit Age (Months)',
                            yaxis_title='Years at Job',
                            zaxis_title='Default Probability',
                            camera=dict(
                                eye=dict(x=1.5, y=1.5, z=0.5)
                            )
                        ),
                        height=600
                    )
                    
                    # Create animation frames
                    frames = [go.Frame(
                        layout=dict(
                            scene_camera=dict(
                                eye=dict(
                                    x=np.cos(np.radians(i)) * 2,
                                    y=np.sin(np.radians(i)) * 2,
                                    z=0.5
                                )
                            )
                        ),
                        name=f'frame{i}'
                    ) for i in range(0, 360, 15)]  # Reduced frame count for performance
                    
                    fig.frames = frames
                    
                    # Add play button
                    fig.update_layout(
                        updatemenus=[{
                            "type": "buttons",
                            "buttons": [
                                {
                                    "label": "▶️ Play",
                                    "method": "animate",
                                    "args": [None, {"frame": {"duration": 100, "redraw": True}}]
                                },
                                {
                                    "label": "⏹️ Stop",
                                    "method": "animate",
                                    "args": [[None], {"frame": {"duration": 0, "redraw": True}}]
                                }
                            ]
                        }]
                    )
                    
                    # Display the figure
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error generating surface animation: {str(e)}")
            else:
                st.warning("Required columns for surface plot not found") 
    
    else:
        st.warning("No data available for the selected filters")

# Footer with animation
# Footer
st.markdown("---")
st.markdown("### 🙏 Thank you for using our dashboard!")
st_lottie(loan_risk_anim, height=250)

# Footer Text
st.markdown("""
<div style='text-align: center; color: gray; font-size: 13px;'>
    Made with 💙 by Ankit Yadav | Powered by Streamlit & Lottie
</div>
""", unsafe_allow_html=True)
st.markdown("""
<div class="footer">
    <p style="font-size:1rem; font-weight:bold;">"Data is the new oil of the digital economy"</p>
    <p style="font-size:0.8rem;">© 2025 LoanRisk AI | Smart Credit Analytics</p>
</div>
""", unsafe_allow_html=True)

