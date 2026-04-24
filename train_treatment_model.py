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

print("📥 Training Treatment Model (Random Split)...")
df = pd.read_csv("OSMI_Mega_Longitudinal_Data.csv")
df['Survey_Year'] = "Year_" + df['Survey_Year'].astype(str)

# 1. Target Mapping
# 1. Target Mapping (FIXED)
target_col = 'Target_Sought_Treatment'
df = df.dropna(subset=[target_col])

# The CSV already uses 1 and 0! We just force it to integer to be safe.
df[target_col] = df[target_col].astype(int)

# 2. Random Stratified Split
X = df.drop(columns=[target_col])
y = df[target_col]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 3. Features & Preprocessing (Work_Interfere becomes a feature here)
cat_cols = ['Gender', 'Family_History', 'Company_Size', 'Tech_Company',
            'Wellness_Program', 'Anonymity_Protected', 'Leave_Difficulty',
            'Benefits', 'Care_Options', 'Survey_Year', 'Work_Interfere']
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
print(f"🔥 Treatment Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=["Will Not Seek (0)", "Will Seek (1)"]))

os.makedirs("models", exist_ok=True)
joblib.dump(pipeline, "models/treatment_model.joblib")
print("✅ Saved to models/treatment_model.joblib")