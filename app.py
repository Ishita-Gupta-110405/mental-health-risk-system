from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
import joblib
import shap
import os
import uvicorn

app = FastAPI(title="MHRD: Dual-Model Mental Health System")
templates = Jinja2Templates(directory="templates")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading Dual Models...")
risk_pipeline = joblib.load("models/temporal_model.joblib")
treatment_pipeline = joblib.load("models/treatment_model.joblib")

risk_model = risk_pipeline.named_steps['model']
risk_preprocessor = risk_pipeline.named_steps['preprocessor']
risk_explainer = shap.TreeExplainer(risk_model)

class SurveyData(BaseModel):
    Age: int
    Gender: str
    Family_History: str
    Company_Size: str
    Tech_Company: str
    Wellness_Program: str
    Anonymity_Protected: str
    Leave_Difficulty: str
    Benefits: str
    Care_Options: str
    Survey_Year: str
    Target_Sought_Treatment: str 
    Work_Interfere: str          
    Comments: str = ""

@app.get("/", response_class=HTMLResponse)
async def serve_ui(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(data: SurveyData):
    input_dict = data.dict()
    
    # 🐛 CRITICAL FIX: The pipeline was trained on integers 1 and 0 for treatment.
    # We must convert the frontend "Yes"/"No" to integers, otherwise the model ignores it!
    if input_dict['Target_Sought_Treatment'] == "Yes":
        input_dict['Target_Sought_Treatment'] = 1
    else:
        input_dict['Target_Sought_Treatment'] = 0

    df = pd.DataFrame([input_dict])
    
    # ==========================================
    # 1. RISK PREDICTION & CONFIDENCE
    # ==========================================
    risk_probs = risk_pipeline.predict_proba(df)[0]
    risk_pred = 1 if risk_probs[1] > 0.5 else 0
    risk_conf = float(risk_probs[1] if risk_pred == 1 else risk_probs[0])
    risk_label = "High Risk of Interference" if risk_pred == 1 else "Low Risk of Interference"
    
    # ==========================================
    # 2. TREATMENT PREDICTION & CONFIDENCE
    # ==========================================
    treat_probs = treatment_pipeline.predict_proba(df)[0]
    treat_pred = 1 if treat_probs[1] > 0.5 else 0
    treat_conf = float(treat_probs[1] if treat_pred == 1 else treat_probs[0])
    treat_label = "Likely to Seek Treatment" if treat_pred == 1 else "Unlikely to Seek Treatment"

    # ==========================================
    # 3. SHAP EXPLAINABILITY (Mapped to Features)
    # ==========================================
    transformed_features = risk_preprocessor.transform(df)
    shap_vals = risk_explainer.shap_values(transformed_features)
    base_shap = shap_vals[risk_pred][0].tolist() if isinstance(shap_vals, list) else shap_vals[0].tolist()
    
    feature_names = risk_preprocessor.get_feature_names_out()
    shap_dict = [{"feature": f.split('__')[-1], "value": float(s)} for f, s in zip(feature_names, base_shap)]
    top_shap = sorted(shap_dict, key=lambda x: abs(x["value"]), reverse=True)[:8]

    # ==========================================
    # 4. TIME VARIANCE SIMULATION (All 4 Years)
    # ==========================================
    years = ["Year_2014", "Year_2016", "Year_2020", "Year_2021"]
    year_labels = ["2014", "2016", "2020", "2021"]
    trend_probs = []
    
    for year in years:
        temp_df = df.copy()
        temp_df['Survey_Year'] = year
        prob = risk_pipeline.predict_proba(temp_df)[0][1] 
        trend_probs.append(float(prob))

    return {
        "risk_prediction": risk_label,
        "risk_confidence": risk_conf,
        "treatment_prediction": treat_label,
        "treatment_confidence": treat_conf,
        "top_shap": top_shap,
        "temporal_trend": {
            "years": year_labels,
            "probabilities": trend_probs
        }
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)