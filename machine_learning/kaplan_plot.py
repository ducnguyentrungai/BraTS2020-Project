import pandas as pd
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

# Đọc file CSV
df = pd.read_csv("/work/cuc.buithi/brats_challenge/test/survival_info_filled.csv")
# Lọc dữ liệu có loại phẫu thuật rõ ràng (GTR, STR)
df_km = df[df['Extent_of_Resection_Encoder'].isin([1, 0])]

# Tạo đối tượng Kaplan-Meier
kmf = KaplanMeierFitter()

# Khởi tạo biểu đồ
group_mapping = {
    1: 'GTR',
    0: 'STR'
}

plt.figure(figsize=(10, 6))
for code, label in group_mapping.items():
    sub_df = df[df['Extent_of_Resection_Encoder'] == code]
    kmf.fit(sub_df['Survival_days'], event_observed=[1]*len(sub_df), label=label)
    kmf.plot_survival_function()

plt.title("Kaplan-Meier Curve theo mã hóa loại phẫu thuật")
plt.xlabel("Số ngày sống")
plt.ylabel("Xác suất sống còn")
plt.grid(True)
plt.legend(title="Loại phẫu thuật")
plt.tight_layout()
plt.savefig("km_curve_encoded.png", dpi=300)
plt.show()