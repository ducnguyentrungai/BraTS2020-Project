import pandas as pd
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
import matplotlib.pyplot as plt

# Đọc file CSV
df = pd.read_csv("/work/cuc.buithi/brats_challenge/test/survival_info_filled.csv")

# Lọc GTR (1) và STR (0)
df = df[df['Extent_of_Resection_Encoder'].isin([0, 1])]

# Tách hai nhóm
df_gtr = df[df['Extent_of_Resection_Encoder'] == 1]
df_str = df[df['Extent_of_Resection_Encoder'] == 0]

# Log-rank test
result = logrank_test(
    df_gtr['Survival_days'],
    df_str['Survival_days'],
    event_observed_A=[1]*len(df_gtr),
    event_observed_B=[1]*len(df_str)
)
p_value = result.p_value

# Kaplan-Meier plot
kmf = KaplanMeierFitter()
plt.figure(figsize=(10, 6))

for code, label in {1: "GTR", 0: "STR"}.items():
    sub_df = df[df['Extent_of_Resection_Encoder'] == code]
    kmf.fit(sub_df["Survival_days"], event_observed=[1]*len(sub_df), label=label)
    kmf.plot_survival_function()

plt.title(f"Kaplan-Meier Curve (Log-Rank p-value = {p_value:.4f})")
plt.xlabel("Số ngày sống")
plt.ylabel("Xác suất sống còn")
plt.legend(title="Loại phẫu thuật")
plt.grid(True)
plt.tight_layout()
plt.savefig("km_logrank_curve.png", dpi=300)
plt.show()
