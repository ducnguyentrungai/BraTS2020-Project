import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

def classify_survival_days_kmeans(df: pd.DataFrame, n_clusters: int = 3, col: str = 'Survival_days'):
    df = df.copy()
    
    # Chỉ xử lý các dòng có giá trị sống sót hợp lệ
    valid_mask = df[col].notna()
    survival_values = df.loc[valid_mask, [col]]

    # Fit KMeans
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
    cluster_labels = km.fit_predict(survival_values)

    # Sắp xếp lại label theo giá trị trung tâm tăng dần
    centers = km.cluster_centers_.flatten()
    sorted_label_map = {old: new for new, old in enumerate(np.argsort(centers))}
    survival_classes = pd.Series(cluster_labels).map(sorted_label_map).values

    # Gán lại vào DataFrame
    df.loc[valid_mask, 'Survival_Class'] = survival_classes.astype(int)
    print(df.head(5))

    # In biểu thời gian sống của từng mức độ
    print("📊 Survival Class Statistics:")
    for cls in sorted(df['Survival_Class'].dropna().unique()):
        group = df[df['Survival_Class'] == cls][col]
        print(f"Class {int(cls)}: {len(group)} samples")
        print(f"  Min  = {group.min():.0f} days")
        print(f"  Max  = {group.max():.0f} days")
        print(f"  Mean = {group.mean():.1f} days\n")

    return df


if __name__ == "__main__":
    df = pd.read_csv("full_info_suvival.csv")
    mapping = {'GTR':float(1), 'STR': float(0)}
    df['Extent_of_Resection_Encode'] = df['Extent_of_Resection'].map(mapping)
    classify_survival_days_kmeans(df=df)
    # print(df.head(5))