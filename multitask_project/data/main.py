import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv('suvivaldays_info.csv', index_col=0)
    df_new = df.copy()
    df_new["Survival_Class_Binary"] = df_new["Survival_Class"].apply(lambda x: 0 if x == 0 else 1)
    df_new.drop(['Survival_Class', "Extent_of_Resection_Encode"], axis=1,inplace=True)
    print(df_new.head())
    df_new.to_csv("suvivaldays_binary.csv", index=False)
    