import joblib
import shap
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re

print("Loading model and data...")
# Load the leakage-free model
risk_model = joblib.load(r"models\temporal_model.joblib")

# Load and prep the data
df_raw = pd.read_csv("OSMI_Mega_Longitudinal_Data.csv")
df_raw['Survey_Year'] = "Year_" + df_raw['Survey_Year'].astype(str)

def parse_work_interfere(val):
    val_str = str(val).lower()
    if 'never' in val_str: return 0
    if 'rarely' in val_str: return 0
    if 'sometimes' in val_str: return 1
    if 'often' in val_str: return 1
    return np.nan

df_risk = df_raw.copy()
df_risk['Work_Interfere'] = df_risk['Work_Interfere'].apply(parse_work_interfere)
df_risk = df_risk.dropna(subset=['Work_Interfere'])
test_years = ['Year_2020', 'Year_2021']
test_df = df_risk[df_risk['Survey_Year'].isin(test_years)]

# Crucial: Drop the leaked column!
X_test = test_df.drop(columns=['Work_Interfere', 'Target_Sought_Treatment'], errors='ignore')

print("Calculating SHAP values...")
# Transform data and get SHAP values
preprocessor = risk_model.named_steps['preprocessor']
transformed_test = np.array(preprocessor.transform(X_test))

explainer = shap.TreeExplainer(risk_model.named_steps['model'])
shap_values = explainer.shap_values(transformed_test)
if isinstance(shap_values, list):
    shap_values = shap_values[1]

# Clean up feature names for the chart
raw_names = preprocessor.get_feature_names_out()
clean_names = [re.sub(r'^.*?__', '', f) for f in raw_names]
clean_names = [f"Comment_Topic_{''.join(filter(str.isdigit, n))}" if 'truncatedsvd' in n.lower() else n for n in clean_names]

# Generate and save the plot
plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values, features=transformed_test, feature_names=clean_names, show=False)
plt.tight_layout()
plt.savefig("shap.png", dpi=300, bbox_inches='tight')
print("✅ New shap.png saved successfully!")