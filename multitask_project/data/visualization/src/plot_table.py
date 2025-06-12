import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, '..', 'images')
os.makedirs(IMAGE_DIR, exist_ok=True)

if __name__ == "__main__":

    # Load dữ liệu
    df = pd.read_csv("/work/cuc.buithi/brats_challenge/code/multitask_project/data/suvivaldays_info.csv")

    # Chọn các feature liên quan
    # features = ['Age', 'tumor_pct', 'ncr_net_pct',
    #             'ed_pct', 'et_pct', 'Extent_of_Resection',]
    # target = 'Survival_Class'
    
    features = ['Age', 'tumor_pct', 'ncr_net_pct',
                'ed_pct', 'et_pct', 'Extent_of_Resection',]
    target = 'Survival_Class'


    n_cols = 3
    n_rows = 2
    # Tăng font chữ mặc định
    plt.rcParams.update({'font.size': 14})

    plt.figure(figsize=(6 * n_cols, 5 * n_rows))  # tăng kích thước chung

    for i, feature in enumerate(features, 1):
        plt.subplot(n_rows, n_cols, i)
        
        if df[feature].dtype == 'object':
            sns.countplot(data=df, x=feature, hue=target)
            plt.xticks(rotation=15)
        else:
            sns.boxplot(data=df, x=target, y=feature, hue=target, palette="Set2", legend=False)
        
        plt.title(f"{feature} vs {target}")
        plt.xlabel("")
        plt.ylabel("")

    # Tăng khoảng cách giữa các biểu đồ
    plt.subplots_adjust(hspace=0.4, wspace=0.3)

    plt.savefig(os.path.join(IMAGE_DIR,'feature_vs_survival_class.png'))
    plt.show()