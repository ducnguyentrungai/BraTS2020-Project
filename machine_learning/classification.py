import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

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

# Xem phân phối nhãn
print("🔍 Class Distribution:")
print(df["Survival_Class"].value_counts(normalize=True))

# Tạo X, y
features = ['Age', 'tumor_volume', 'ncr_net_volume', 'ed_volume', 'et_volume',
            'brain_volume', 'tumor_pct', 'ncr_net_pct', 'ed_pct', 'et_pct',
            'Extent_of_Resection_Encoder']
X = df[features]
y = df["Survival_Class"]

# Train/test split + scale
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Khởi tạo mô hình
models = {
    "Logistic_Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
    "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
    "Gradient_Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(probability=True, class_weight='balanced'),
    "KNN": KNeighborsClassifier(n_neighbors=5)
}

# Tạo thư mục lưu kết quả
os.makedirs("classification_reports", exist_ok=True)

# Huấn luyện và lưu từng báo cáo
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, preds)
    conf = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds, zero_division=0)

    # Ghi kết quả vào file
    with open(f"classification_reports/{name}_report.txt", "w") as f:
        f.write(f"Model: {name}\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(conf))
        f.write("\n\nClassification Report:\n")
        f.write(report)

    print(f"✅ Saved report for {name}")