import os
import numpy as np
import pandas as pd

def parse_survival(val):
    try:
        if isinstance(val, str) and 'ALIVE' in val:
            return float(val.split('(')[1].split()[0])  # lấy số trong ngoặc
        else:
            return float(val)
    except:
        return np.nan
    
if __name__ == "__main__":
    out_path = "/work/cuc.buithi/brats_challenge/subdata"
    df = pd.read_csv('/work/cuc.buithi/brats_challenge/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData/survival_info.csv')
    df["Survival_days"] = df["Survival_days"].apply(parse_survival).astype(np.float32)
    mapping = {"GTR": 0, "STR": 1}
    df['Extent_of_Resection_encoder'] = df['Extent_of_Resection'].map(mapping)
    df_not_na = df[df['Extent_of_Resection_encoder'].notna()].copy()
    df_not_na['Age'] = df_not_na['Age'].astype(np.float32)
    df_not_na.to_csv(os.path.join(out_path, 'data_info.csv'), index=False)