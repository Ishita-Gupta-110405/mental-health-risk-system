import warnings
warnings.filterwarnings("ignore")

import re
import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import numpy as np
import joblib
import shap

# ==============================
# ML PIPELINE IMPORTS (Required for joblib to unpickle)
# ==============================
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier

# ==============================
# DATA PREPARATION LOGIC
# ==============================
def parse_work_interfere(val):
    val_str = str(val).lower()
    if 'never' in val_str: return 0
    if 'rarely' in val_str: return 0
    if 'sometimes' in val_str: return 1
    if 'often' in val_str: return 1
    return np.nan

def preprocess_dataframe(df):
    df = df.copy()
    df['Survey_Year'] = "Year_" + df['Survey_Year'].astype(str)
    df['Work_Interfere'] = df['Work_Interfere'].apply(parse_work_interfere)
    df = df.dropna(subset=['Work_Interfere'])
    return df

# ==============================
# FASTAPI SETUP
# ==============================
app = FastAPI()
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# LOAD PRE-TRAINED MODELS & EXPLAINERS
# ==============================
# Dynamically get the absolute path of the directory where app.py lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    risk_model = joblib.load(os.path.join(BASE_DIR, "models", "temporal_model.joblib"))
    treatment_model = joblib.load(os.path.join(BASE_DIR, "models", "treatment_model.joblib"))
    
    explainer = shap.TreeExplainer(risk_model.named_steps['model'])
    explainer_treat = shap.TreeExplainer(treatment_model.named_steps['model'])
    print("✅ Models and SHAP Explainers loaded successfully.")
except FileNotFoundError as e:
    print(f"❌ Error: Model files not found. Detailed path error: {e}")

# ==============================
# STARTUP METRICS EVALUATION (Inference Only)
# ==============================
print("Evaluating models dynamically on test set...")
try:
    df = pd.read_csv("OSMI_Mega_Longitudinal_Data.csv")
    df = preprocess_dataframe(df)

    test_years = ['Year_2020', 'Year_2021']
    
    # Risk model eval
    test_df = df[df['Survey_Year'].isin(test_years)]
    if not test_df.empty:
        X_test_risk = test_df.drop(columns=['Work_Interfere', 'Target_Sought_Treatment'], errors='ignore')
        y_test_risk = test_df['Work_Interfere']
        risk_pred = risk_model.predict(X_test_risk)
        risk_proba = risk_model.predict_proba(X_test_risk)[:, 1]
        risk_accuracy = accuracy_score(y_test_risk, risk_pred)
        risk_auc = roc_auc_score(y_test_risk, risk_proba)
    else:
        risk_accuracy, risk_auc = 0.0, 0.0

    # Treatment model eval
    X = df.drop(columns=['Target_Sought_Treatment', 'Work_Interfere'], errors='ignore')
    if 'Target_Sought_Treatment' in df.columns:
        target_series = df['Target_Sought_Treatment']
        if target_series.dtype not in ['int64', 'float64']:
            target_series = target_series.astype(str).str.strip().str.lower().map({'yes': 1, 'no': 0})
        
        valid_idx = target_series.dropna().index
        X_valid = X.loc[valid_idx]
        y_valid = target_series.loc[valid_idx].astype(int)
        
        X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(
            X_valid, y_valid, test_size=0.2, stratify=y_valid, random_state=42
        )
        treat_pred = treatment_model.predict(X_test_t)
        treat_proba = treatment_model.predict_proba(X_test_t)[:, 1]
        treat_accuracy = accuracy_score(y_test_t, treat_pred)
        treat_auc = roc_auc_score(y_test_t, treat_proba)
    else:
        treat_accuracy, treat_auc = 0.0, 0.0
except Exception as e:
    print(f"Metrics evaluation error: {e}")
    risk_accuracy = risk_auc = treat_accuracy = treat_auc = 0.0

# ==============================
# API ENDPOINTS
# ==============================
@app.get("/", response_class=HTMLResponse)
def serve_ui(request: Request):
    return templates.TemplateResponse(request, "index.html", {"request": request})

@app.get("/model_performance")
def model_performance():
    return {
        "risk_accuracy": round(risk_accuracy, 3),
        "risk_auc": round(risk_auc, 3),
        "treatment_accuracy": round(treat_accuracy, 3),
        "treatment_auc": round(treat_auc, 3)
    }

@app.post("/predict")
def predict(data: dict):
    df_full = pd.DataFrame([data])
    expected_columns = [
        'Age', 'Gender', 'Family_History', 'Company_Size', 'Tech_Company',
        'Wellness_Program', 'Anonymity_Protected', 'Leave_Difficulty',
        'Benefits', 'Care_Options', 'Comments', 'Survey_Year',
        'Work_Interfere'
    ]
    df_full = df_full.reindex(columns=expected_columns, fill_value=np.nan)

    df_risk = df_full.drop(columns=['Work_Interfere'], errors='ignore')
    df_treatment = df_full.drop(columns=['Target_Sought_Treatment'], errors='ignore')

    # Base Predictions
    risk_pred = int(risk_model.predict(df_risk)[0])
    risk_prob = float(risk_model.predict_proba(df_risk)[0][1])

    treatment_pred = int(treatment_model.predict(df_treatment)[0])
    treatment_prob = float(treatment_model.predict_proba(df_treatment)[0][1])

    # Extract target year from user input to cap the graph
    target_year_str = str(data.get('Survey_Year', '2099'))
    try:
        # e.g., "Year_2020" -> 2020
        max_year = int(re.search(r'\d+', target_year_str).group())
    except AttributeError:
        max_year = 2099 # Fallback if regex fails

    top_features = []
    year_impact = {}
    base_value = 0.0

    try:
        # -------------------------------------------------------------------
        # 1. SHAP Features (From Risk Model)
        # -------------------------------------------------------------------
        transformed_risk = risk_model.named_steps['preprocessor'].transform(df_risk)
        transformed_risk = np.array(transformed_risk)
        shap_values_risk = explainer.shap_values(transformed_risk)
        
        if isinstance(shap_values_risk, list):
            shap_values_risk = shap_values_risk[1]
        if shap_values_risk.ndim == 2:
            shap_values_risk = shap_values_risk[0]

        base_val = explainer.expected_value
        base_value = float(base_val[1] if isinstance(base_val, (list, np.ndarray)) and len(base_val) > 1 else base_val)

        raw_feature_names = risk_model.named_steps['preprocessor'].get_feature_names_out()
        
        clean_names = []
        for f in raw_feature_names:
            name = re.sub(r'^.*?__', '', f)
            if 'truncatedsvd' in name.lower():
                num = ''.join(filter(str.isdigit, name))
                name = f"Comment_Topic_{num}"
            clean_names.append(name)
        
        shap_pairs = list(zip(clean_names, shap_values_risk))
        shap_pairs_sorted = sorted(shap_pairs, key=lambda x: abs(x[1]), reverse=True)
        top_features = [{"feature": f, "impact": float(v)} for f, v in shap_pairs_sorted[:8]]
        
        # -------------------------------------------------------------------
        # 2. TEMPORAL SHAP DATA (From Treatment Model, Capped at Input Year)
        # -------------------------------------------------------------------
        transformed_treat = treatment_model.named_steps['preprocessor'].transform(df_treatment)
        transformed_treat = np.array(transformed_treat)
        shap_values_treat = explainer_treat.shap_values(transformed_treat)
        
        if isinstance(shap_values_treat, list):
            shap_values_treat = shap_values_treat[1]
        if shap_values_treat.ndim == 2:
            shap_values_treat = shap_values_treat[0]
            
        treat_feature_names = treatment_model.named_steps['preprocessor'].get_feature_names_out()
        treat_clean_names = [re.sub(r'^.*?__', '', f) for f in treat_feature_names]
        treat_shap_pairs = list(zip(treat_clean_names, shap_values_treat))

        # Extract years and values
        raw_years = {}
        for f, v in treat_shap_pairs:
            if "Survey_Year" in f:
                y_str = f.replace('Survey_Year_Year_', '').replace('Survey_Year_', '')
                raw_years[y_str] = float(v)
        
        # Filter strictly to years <= the user's dropdown selection
        filtered_years = {}
        for y_str, v in raw_years.items():
            if y_str.isdigit() and int(y_str) <= max_year:
                filtered_years[y_str] = v
                
        # Sort chronologically
        year_impact = {k: filtered_years[k] for k in sorted(filtered_years.keys())}

    except Exception as e:
        print(f"SHAP error: {e}")

    return {
        "risk_prediction": risk_pred,
        "risk_confidence": risk_prob,
        "risk_label": "High Work Interference Risk" if risk_pred == 1 else "Low Work Interference Risk",
        "treatment_prediction": treatment_pred,
        "treatment_label": "Likely to Seek Treatment" if treatment_pred == 1 else "Unlikely to Seek Treatment",
        "treatment_confidence": treatment_prob,
        "top_features": top_features,
        "year_impact": year_impact,
        "base_value": base_value
    }