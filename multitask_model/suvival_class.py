import pandas as pd
from sklearn.model_selection import train_test_split

# def map_survival_class(days):
#     if days <= 300:
#         return 0  # Short
#     elif days <= 800:
#         return 1  # Medium
#     else:
#         return 2  # Long
    

# df = pd.read_csv('/work/cuc.buithi/brats_challenge/code/multitask_model/data/survival_info_filled.csv')

# df['Survival_Class'] = df['Survival_days'].apply(map_survival_class)
# df.to_csv('/work/cuc.buithi/brats_challenge/code/multitask_model/data/survival.csv')

if __name__ == '__main__':
    df = pd.read_csv('/work/cuc.buithi/brats_challenge/code/multitask_model/data/survival.csv')
    # print(df['Survival_Class'].value_counts().sort_index())

    # Giả sử bạn đã có cột Survival_Class
    train_df, test_df = train_test_split(
        df,
        test_size=0.1,             
        stratify=df['Survival_Class'],  # Bảo đảm tỷ lệ giữa các class giống nhau ở train/test
        random_state=42
    )

    # Xác nhận phân bố lớp
    print("Train class counts:\n", train_df['Survival_Class'].value_counts().sort_index())
    print("Test class counts:\n", test_df['Survival_Class'].value_counts().sort_index())
    print(train_df.head())
    train_df.to_csv('/work/cuc.buithi/brats_challenge/code/multitask_model/data/train.csv', index=False)
    test_df.to_csv('/work/cuc.buithi/brats_challenge/code/multitask_model/data/test.csv', index=False)
    