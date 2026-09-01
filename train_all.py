import os
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier

def create_model_pipeline():
    cat_cols = ['Gender', 'Family_History', 'Company_Size', 'Tech_Company',
                'Wellness_Program', 'Anonymity_Protected', 'Leave_Difficulty',
                'Benefits', 'Care_Options', 'Survey_Year']
    num_cols = ['Age']
    text_col = 'Comments'

    nlp_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=1000)),
        ('svd', TruncatedSVD(n_components=10, random_state=42))
    ])

    preprocessor = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', IterativeImputer(random_state=42)),
            ('scaler', StandardScaler())
        ]), num_cols),

        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), cat_cols),

        ('text', nlp_pipeline, text_col)
    ])

    model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        class_weight='balanced',
        random_state=42,
        verbose=-1
    )

    pipeline = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42, k_neighbors=3)),
        ('model', model)
    ])
    
    return pipeline

def parse_work_interfere(val):
    """Robust substring matching to prevent data loss in later years."""
    val_str = str(val).lower()
    if 'never' in val_str: return 0
    if 'rarely' in val_str: return 0
    if 'sometimes' in val_str: return 1
    if 'often' in val_str: return 1
    return np.nan

print("📥 Loading dataset...")
df_raw = pd.read_csv("OSMI_Mega_Longitudinal_Data.csv")
df_raw['Survey_Year'] = "Year_" + df_raw['Survey_Year'].astype(str)
os.makedirs("models", exist_ok=True)

# ==============================
# TRAIN RISK MODEL 
# ==============================
print("\n🚀 TRAINING RISK MODEL")
df_risk = df_raw.copy()
df = df_risk.drop(columns=['Target_Sought_Treatment'], errors='ignore')

# Apply the robust mapping to rescue 2017+ data
df_risk['Work_Interfere'] = df_risk['Work_Interfere'].apply(parse_work_interfere)
df_risk = df_risk.dropna(subset=['Work_Interfere'])

# Now that the data exists, include it in training
train_years = ['Year_2014', 'Year_2016', 'Year_2017', 'Year_2018', 'Year_2019']
test_years  = ['Year_2020', 'Year_2021']

train_df = df_risk[df_risk['Survey_Year'].isin(train_years)]
test_df  = df_risk[df_risk['Survey_Year'].isin(test_years)]

X_train_risk = train_df.drop(columns=['Work_Interfere', 'Target_Sought_Treatment'], errors='ignore')
y_train_risk = train_df['Work_Interfere']
X_test_risk = test_df.drop(columns=['Work_Interfere', 'Target_Sought_Treatment'], errors='ignore')
y_test_risk = test_df['Work_Interfere']

risk_pipeline = create_model_pipeline()
risk_pipeline.fit(X_train_risk, y_train_risk)
joblib.dump(risk_pipeline, "models/temporal_model.joblib")
print("✅ Saved -> models/temporal_model.joblib")

# ==============================
# TRAIN TREATMENT MODEL
# ==============================
print("\n🚀 TRAINING TREATMENT MODEL")
df_treat = df_raw.copy()
target_col = 'Target_Sought_Treatment'

if df_treat[target_col].dtype in ['int64', 'float64']:
    df_treat = df_treat.dropna(subset=[target_col])
    df_treat[target_col] = df_treat[target_col].astype(int)
else:
    df_treat[target_col] = df_treat[target_col].astype(str).str.strip().str.lower()
    df_treat[target_col] = df_treat[target_col].map({'yes': 1, 'no': 0})
    df_treat = df_treat.dropna(subset=[target_col])

X_treat = df_treat.drop(columns=[target_col, 'Work_Interfere'], errors='ignore')
y_treat = df_treat[target_col]

X_train_treat, X_test_treat, y_train_treat, y_test_treat = train_test_split(
    X_treat, y_treat, test_size=0.2, stratify=y_treat, random_state=42
)

treat_pipeline = create_model_pipeline()
treat_pipeline.fit(X_train_treat, y_train_treat)
joblib.dump(risk_pipeline, "models/temporal_model.joblib")
joblib.dump(treat_pipeline, "models/treatment_model.joblib")
print("✅ Saved -> models/treatment_model.joblib")
print("🎉 Training complete!")