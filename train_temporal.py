import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
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

print("📥 Training Risk Model (Temporal Split)...")
df = pd.read_csv("OSMI_Mega_Longitudinal_Data.csv")

# 1. Clean and Binarize Target
target_col = 'Work_Interfere'
df = df.dropna(subset=[target_col])
severity_map = {'Never': 0, 'Rarely': 0, 'Sometimes': 1, 'Often': 1}
df[target_col] = df[target_col].map(severity_map)

# 2. Temporal Split
train_df = df[df['Survey_Year'].isin([2014, 2016])].copy()
test_df = df[df['Survey_Year'].isin([2020, 2021])].copy()

train_df['Survey_Year'] = "Year_" + train_df['Survey_Year'].astype(str)
test_df['Survey_Year'] = "Year_" + test_df['Survey_Year'].astype(str)

X_train = train_df.drop(columns=[target_col])
y_train = train_df[target_col]
X_test = test_df.drop(columns=[target_col])
y_test = test_df[target_col]

# 3. Features & Preprocessing
cat_cols = ['Gender', 'Family_History', 'Company_Size', 'Tech_Company',
            'Wellness_Program', 'Anonymity_Protected', 'Leave_Difficulty',
            'Benefits', 'Care_Options', 'Survey_Year', 'Target_Sought_Treatment']
num_cols = ['Age']
text_col = 'Comments'

nlp_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=1000, ngram_range=(1, 2))),
    ('svd', TruncatedSVD(n_components=10, random_state=42))
])

preprocessor = ColumnTransformer([
    ('num', Pipeline([('imputer', IterativeImputer(random_state=42)), ('scaler', StandardScaler())]), num_cols),
    ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_cols),
    ('text', nlp_pipeline, text_col)
])

# 4. Pipeline & Training
pipeline = ImbPipeline([
    ('preprocessor', preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('model', LGBMClassifier(n_estimators=200, learning_rate=0.05, max_depth=-1, num_leaves=31, random_state=42, verbose=-1))
])

pipeline.fit(X_train, y_train)

# 5. Evaluate & Save
y_pred = pipeline.predict(X_test)
print(f"🔥 Temporal Risk Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=["Low Risk (0)", "High Risk (1)"]))

os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/temporal_model.joblib")
print("✅ Saved to models/temporal_model.joblib")