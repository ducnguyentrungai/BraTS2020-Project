import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, FunctionTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer

# Load data
df = pd.read_csv("/work/cuc.buithi/brats_challenge/test/survival_info_filled.csv")

# Tạo nhãn nhóm sống sót
def categorize_survival(days):
    if days <= 180:
        return "Short"
    elif days <= 730:
        return "Medium"
    else:
        return "Long"

df["Survival_Class"] = df["Survival_days"].apply(categorize_survival)

# Tạo đặc trưng mới
df["tumor_to_brain_ratio"] = df["tumor_volume"] / (df["brain_volume"] + 1e-8)
df["active_tumor_ratio"] = df["et_volume"] / (df["tumor_volume"] + 1e-8)

# Chọn features
features = ['Age', 'tumor_volume', 'ncr_net_volume', 'ed_volume', 'et_volume',
            'brain_volume', 'tumor_pct', 'ncr_net_pct', 'ed_pct', 'et_pct',
            'Extent_of_Resection_Encoder', 'tumor_to_brain_ratio', 'active_tumor_ratio']
X = df[features]
y = df["Survival_Class"]

# Chia train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# Log-transform các cột skew
log_cols = ['tumor_volume', 'ncr_net_volume', 'ed_volume', 'et_volume', 'brain_volume']
log_transformer = FunctionTransformer(np.log1p, validate=True)

preprocessor = ColumnTransformer(
    transformers=[
        ('log', log_transformer, log_cols),
        ('scale', StandardScaler(), features)  # scale toàn bộ
    ],
    remainder='passthrough'
)

# SMOTE để cân bằng
smote = SMOTE(random_state=42)
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
X_train_bal, y_train_bal = smote.fit_resample(X_train_processed, y_train)

# Huấn luyện với CatBoost
model = CatBoostClassifier(verbose=0, random_state=42)
model.fit(X_train_bal, y_train_bal)

# Dự đoán và đánh giá
preds = model.predict(X_test_processed)
acc = accuracy_score(y_test, preds)
f1 = f1_score(y_test, preds, average='macro')
conf = confusion_matrix(y_test, preds)
report = classification_report(y_test, preds, zero_division=0)


with open("classification_reports/CatBoost_SMOTE_report.txt", "w") as f:
    f.write(f"Accuracy: {acc:.4f}\n")
    f.write(f"F1-score (macro): {f1:.4f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(conf))
    f.write("\n\nClassification Report:\n")
    f.write(report)

print("✅ Report saved to classification_reports/CatBoost_SMOTE_report.txt")
