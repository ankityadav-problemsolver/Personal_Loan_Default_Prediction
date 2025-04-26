# model_metrics.py

import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve,
    confusion_matrix,
    classification_report
)

# --- Feature Engineering ---
def engineer_features(df, monthly_rate=0.01, loan_term=12):
    df = df.copy()

    # Original feature creation
    df['EMI'] = (df['LOAN'] * monthly_rate * (1 + monthly_rate)**loan_term) / ((1 + monthly_rate)**loan_term - 1)

    # Binned features
    df['YOJ_BINNED'] = pd.cut(df['YOJ'], 
                              bins=[-1, 2, 5, 10, 20, 40], 
                              labels=['0-2', '2-5', '5-10', '10-20', '20+'])
    df['CLAGE_BINNED'] = pd.cut(df['CLAGE'], 
                                bins=[-1, 60, 120, 180, 240, 500], 
                                labels=['0-5', '5-10', '10-15', '15-20', '20+'])
    
    # Other features
    df['LOAN_TO_VALUE'] = df['LOAN'] / df['VALUE']
    df['LOAN_TO_MORTDUE'] = df['LOAN'] / (df['MORTDUE'] + 1)
    df['DEROG_DELINQ_SUM'] = df['DEROG'] + df['DELINQ']
    df['CLAGE_PER_CLNO'] = df['CLAGE'] / (df['CLNO'] + 1)

    # Convert categorical bins into numeric codes
    cat_cols = ['YOJ_BINNED', 'CLAGE_BINNED']
    for col in cat_cols:
        df[col] = df[col].cat.codes.replace(-1, np.nan)  # Missing bins set to NaN

    return df

# --- Align features ---
def align_features(X, model):
    if hasattr(model, 'feature_names_in_'):
        expected = model.feature_names_in_
        missing = set(expected) - set(X.columns)
        for col in missing:
            X[col] = 0
        X = X[expected]
    return X

# --- ROC Curve ---
def plot_roc(y_true, y_proba, color_scheme):
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc_score = auc(fpr, tpr)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines',
                             name=f'ROC Curve (AUC = {auc_score:.2f})',
                             line=dict(color=color_scheme['primary'], width=3)))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines',
                             name='Random Guessing',
                             line=dict(color=color_scheme['danger'], dash='dash')))
    fig.update_layout(
        xaxis_title='False Positive Rate',
        yaxis_title='True Positive Rate',
        plot_bgcolor=color_scheme['card'],
        paper_bgcolor=color_scheme['background'],
        font=dict(color=color_scheme['text']),
        margin=dict(l=50, r=50, b=50, t=50)
    )
    return fig, auc_score

# --- Precision-Recall Curve ---
def plot_precision_recall(y_true, y_proba, color_scheme):
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=recall, y=precision, mode='lines',
                             name='Precision-Recall Curve',
                             line=dict(color=color_scheme['success'], width=3)))
    fig.update_layout(
        xaxis_title='Recall',
        yaxis_title='Precision',
        plot_bgcolor=color_scheme['card'],
        paper_bgcolor=color_scheme['background'],
        font=dict(color=color_scheme['text']),
        margin=dict(l=50, r=50, b=50, t=50)
    )
    return fig

# --- Confusion Matrix Heatmap ----
def plot_confusion_matrix(y_true, y_pred, color_scheme):
    cm = confusion_matrix(y_true, y_pred)
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted 0', 'Predicted 1'],
        y=['Actual 0', 'Actual 1'],
        colorscale='cividis',  # Updated to 'crest' colorscale
        showscale=True
    ))
    fig.update_layout(
        title='Confusion Matrix',
        plot_bgcolor=color_scheme['card'],
        paper_bgcolor=color_scheme['background'],
        font=dict(color=color_scheme['text']),
        margin=dict(l=50, r=50, b=50, t=50)
    )
    return fig

# --- Classification Report Text ---
def generate_classification_report(y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=False)
    return report
