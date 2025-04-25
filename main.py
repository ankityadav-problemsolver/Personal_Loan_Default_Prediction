import streamlit as st
import joblib
import pandas as pd
import numpy as np
import shap
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import roc_curve, auc
from streamlit_shap import st_shap
import time
import os
from time import strftime
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import numpy as np
from scipy.stats import gaussian_kde



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
        margin: 20px 0;
        background: linear-gradient(135deg, {COLOR_SCHEME['light']}, {COLOR_SCHEME['white']});
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
    with st.form("loan_appraisal"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"#### <span style='color:{COLOR_SCHEME['primary']}'>Borrower Information</span>", unsafe_allow_html=True)
            loan = st.number_input("Loan Amount (USD)", 1000, 1000000, 5000, 
                                 help="The amount of money requested by the borrower")
            value = st.number_input("Property Value (USD)", 10000, 2000000, 250000,
                                  help="Current market value of the property")
            mortdue = st.number_input("Existing Mortgage (USD)", 0, 1000000, 100000,
                                    help="Outstanding mortgage balance on the property")
            debtinc = st.slider("Debt-to-Income Ratio (%)", 0.0, 100.0, 35.0, 0.5,
                              help="Monthly debt payments divided by gross monthly income")
            
        with col2:
            st.markdown(f"#### <span style='color:{COLOR_SCHEME['secondary']}'>Credit History</span>", unsafe_allow_html=True)
            yoj = st.slider("Years at Current Job", 0, 40, 5,
                          help="Duration of employment with current employer")
            derog = st.number_input("Derogatory Reports", 0, 10, 0,
                                  help="Number of major derogatory reports")
            delinq = st.number_input("Delinquent Accounts", 0, 10, 0,
                                   help="Number of delinquent credit lines")
            clage = st.number_input("Oldest Credit Line (Months)", 0, 500, 120,
                                  help="Age of oldest credit line in months")
            ninq = st.number_input("Recent Credit Inquiries", 0, 10, 0,
                                 help="Number of recent credit inquiries")
            
        submitted = st.form_submit_button("Assess Risk", type="primary")
    
    if submitted and model is not None:
        try:
            # Validate inputs
            if value <= 0:
                st.warning("Property value must be positive")
            if loan <= 0:
                st.warning("Loan amount must be positive")
            if mortdue < 0:
                st.warning("Mortgage balance cannot be negative")
            if debtinc < 0 or debtinc > 100:
                st.warning("Debt-to-income ratio must be between 0-100%")
            
            # Create input DataFrame
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
                'REASON': 'DebtCon',  # Default value
                'JOB': 'Other'       # Default value
            }
            
            input_df = pd.DataFrame([input_data])
            
            # Feature engineering to match model training
            input_df['LOAN_TO_VALUE'] = input_df['LOAN'] / input_df['VALUE']
            input_df['DEBT_TO_INCOME'] = input_df['DEBTINC']
            
            # Make prediction
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]
            
            # Display results
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
            try:
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(input_df)
                st.markdown("#### Risk Factor Analysis")
                st_shap(shap.force_plot(
                    explainer.expected_value, 
                    shap_values, 
                    input_df,
                    plot_cmap=[COLOR_SCHEME['success'], COLOR_SCHEME['danger']]
                ))
            except Exception as e:
                st.warning(f"Could not generate SHAP explanation: {str(e)}")
        
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
    
    # Model Insights Section
    if model is not None:
        st.markdown("---")
        st.markdown("## Model Performance Analytics")
        
        # ROC Curve with proper data handling
        if 'BAD' in df.columns:
            try:
                with st.spinner('Calculating model performance metrics...'):
                    X = df.drop('BAD', axis=1)
                    y = df['BAD']
                    
                    # Align columns with model's expected features
                    if hasattr(model, 'feature_names_in_'):
                        available_features = [col for col in model.feature_names_in_ if col in X.columns]
                        X = X[available_features]
                        
                        # Add missing features with default values
                        missing_features = set(model.feature_names_in_) - set(X.columns)
                        for feature in missing_features:
                            X[feature] = 0
                        
                        # Reorder columns to match model expectations
                        X = X[model.feature_names_in_]
                    
                    probas = model.predict_proba(X)[:, 1]
                    fpr, tpr, thresholds = roc_curve(y, probas)
                    roc_auc = auc(fpr, tpr)
                    
                    with st.container():
                        st.markdown("#### Model Discrimination Ability (ROC Curve)")
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=fpr, y=tpr,
                            mode='lines',
                            line=dict(color=COLOR_SCHEME['primary'], width=3),
                            name=f'ROC Curve (AUC = {roc_auc:.2f})'
                        ))
                        fig.add_trace(go.Scatter(
                            x=[0, 1], y=[0, 1],
                            mode='lines',
                            line=dict(color=COLOR_SCHEME['danger'], dash='dash'),
                            name='Random Guessing'
                        ))
                        fig.update_layout(
                            xaxis_title='False Positive Rate',
                            yaxis_title='True Positive Rate',
                            height=500,
                            plot_bgcolor=COLOR_SCHEME['card'],
                            paper_bgcolor=COLOR_SCHEME['background'],
                            font=dict(color=COLOR_SCHEME['text']),
                            margin=dict(l=50, r=50, b=50, t=50),
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
                        st.plotly_chart(fig, use_container_width=True)
                        st.markdown(f"""
                        <div style="color:{COLOR_SCHEME['text']};">
                            <p>The ROC curve shows the trade-off between true positive rate (sensitivity) and false positive rate (1-specificity). 
                            Our model achieves an AUC of <strong>{roc_auc:.2f}</strong>, indicating good discrimination ability. 
                            An AUC of 1.0 represents perfect prediction, while 0.5 represents random guessing.</p>
                        </div>
                        """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error generating ROC curve: {str(e)}")
        
        # Feature Importance with proper handling
        if hasattr(model, 'feature_importances_'):
            try:
                with st.container():
                    st.markdown("#### Key Risk Drivers (Feature Importance)")
                    if hasattr(model, 'feature_names_in_'):
                        features = model.feature_names_in_
                        importances = model.feature_importances_
                    else:
                        features = [f"Feature {i}" for i in range(len(model.feature_importances_))]
                        importances = model.feature_importances_
                    
                    # Create DataFrame and sort
                    feat_imp = pd.DataFrame({
                        'Feature': features,
                        'Importance': importances
                    }).sort_values('Importance', ascending=True)  # Sort for horizontal bar chart
                    
                    # Create horizontal bar chart
                    fig = px.bar(
                        feat_imp.tail(15),  # Show top 15 features
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        color='Importance',
                        color_continuous_scale='Viridis',
                        title=''
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
                            tickfont=dict(color=COLOR_SCHEME['text'])
                        )
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.markdown(f"""
                    <div style="color:{COLOR_SCHEME['text']};">
                        <p>This chart shows which factors most influence the model's risk assessment. 
                        Higher values indicate features that contribute more to predicting loan defaults. 
                        Understanding these drivers helps in making informed lending decisions and explaining model behavior.</p>
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

