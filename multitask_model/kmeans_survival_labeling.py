import pandas as pd
from sklearn.cluster import KMeans

# Đọc dữ liệu
df = pd.read_csv("/work/cuc.buithi/brats_challenge/code/multitask_model/data/survival_info_filled.csv")

# Chọn các đặc trưng để phân cụm
features = df[['Age', 'tumor_volume', 'ncr_net_volume', 'ed_volume', 'et_volume',
               'brain_volume', 'tumor_pct', 'ncr_net_pct', 'ed_pct', 'et_pct']]

# Áp dụng KMeans để phân cụm
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Survival_Cluster'] = kmeans.fit_predict(features)

# Dùng nhãn phân cụm làm label
df['Survival_Label'] = df['Survival_Cluster']

# Lưu ra file mới
df.to_csv("/work/cuc.buithi/brats_challenge/code/multitask_model/data/survival_info_labeled.csv", index=False)

print("Hoàn tất! File mới đã lưu thành 'survival_info_labeled.csv'")
