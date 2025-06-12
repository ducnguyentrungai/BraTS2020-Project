import os
import shutil
import numpy as np
import pandas as pd
import nibabel as nib
from tqdm import tqdm
from pprint import pprint
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from monai.utils import first
from torch.utils.data import DataLoader, WeightedRandomSampler
from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd, ScaleIntensityd,Resized, CastToTyped, ToTensord

class MultitaskDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx].copy()
        
        class_label = item["class_label"]
        tabular = item["tabular"]
        transform_input = {"image": item["image"], "label": item["label"]}

        if self.transform:
            transformed = self.transform(transform_input)
            item['image'] = transformed['image']
            item['label'] = transformed['label']

        if not isinstance(class_label, torch.Tensor):
            class_label = torch.tensor(class_label, dtype=torch.long)
        if not isinstance(tabular, torch.Tensor):
            tabular = torch.tensor(tabular, dtype=torch.float32)

        item["class_label"] = class_label
        item["tabular"] = tabular
        return item


def prepare_data_list(images_path, labels_path, classes_path) -> list:
    df_info = pd.read_csv(classes_path)

    # Chuẩn hóa các đặc trưng tabular
    '''
    
    Brats20ID,Age,Survival_days,Extent_of_Resection,tumor_volume,ncr_net_volume,ed_volume,et_volume,brain_volume,
    tumor_pct,ncr_net_pct,ed_pct,et_pct,Grade,Grade_Encode
    
    '''
    features = ['Age', 'tumor_volume', 'ncr_net_volume', 'ed_volume', 'et_volume', 'brain_volume',
            'tumor_pct', 'ncr_net_pct', 'ed_pct'
            ]

    for col in features:
        df_info[col] = (df_info[col] - df_info[col].min()) / (df_info[col].max() - df_info[col].min())

    list_data = []

    for _, row in df_info.iterrows():
        id_name = row["Brats20ID"]
        case_id = id_name.split('_')[-1]
        image_path = os.path.join(images_path, "image_" + case_id + ".nii.gz")
        label_path = os.path.join(labels_path, "label_" + case_id + ".nii.gz")

        if not os.path.exists(image_path) or not os.path.exists(label_path):
            print('Not found path!')
            continue

        # Tạo vector đặc trưng tabular
        
        features = ['Age', 'tumor_volume', 'ncr_net_volume', 'ed_volume', 'et_volume', 'brain_volume',
            'tumor_pct', 'ncr_net_pct', 'ed_pct', 'Grade_Encode'
            ]
        tabular_features = [
            float(row["Age"]),
            float(row["tumor_volume"]),
            float(row["ncr_net_volume"]),
            float(row["ed_volume"]),
            float(row["et_volume"]),
            float(row["brain_volume"]),
            float(row["tumor_pct"]),
            float(row["ncr_net_pct"]),
            float(row["ed_pct"]),
            float(row["Grade_Encode"]),           
        ]

        list_data.append({
            "image": image_path,
            "label": label_path,
            "class_label": int(row["Survival_days"]),
            "tabular": torch.tensor(tabular_features, dtype=torch.float32)
        })

    return list_data

def get_dataset(data_list:list, transform=None):
    return MultitaskDataset(data_list=data_list, transform=transform)

def get_dataloader(dataset, batch_size: int, num_workers: int, shuffle: bool = False, drop_last: bool = False, use_weighted_sampler: bool = False) -> DataLoader:
    if use_weighted_sampler:
        # Lấy toàn bộ class_label trong dataset
        class_labels = [item['class_label'] for item in dataset.data_list]
        class_counts = [class_labels.count(0), class_labels.count(1)]

        # Tính trọng số ngược lại số lượng mẫu
        total = sum(class_counts)
        class_weights = [total / (len(class_counts) * c) for c in class_counts]
        sample_weights = [class_weights[label] for label in class_labels]

        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

        return DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers, drop_last=drop_last)

    # Nếu không dùng sampler → dùng shuffle như bình thường
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)



if __name__ == "__main__":
    images_path = "/work/cuc.buithi/brats_challenge/data/train_t1_t1ce_t2_flair/imageTr"
    labels_path = "/work/cuc.buithi/brats_challenge/data/train_t1_t1ce_t2_flair/labelTr"
    table_path = "/work/cuc.buithi/brats_challenge/test/fill_data_miss/full_info_suvival.csv"
    transform = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    ScaleIntensityd(keys=["image"]),
    Resized(keys=["image", "label"], spatial_size=(128, 128, 128)),
    ToTensord(keys=["image", "label"]),
    CastToTyped(keys=["label"], dtype=torch.long),
    ])
    out = prepare_data_list(images_path, labels_path, table_path)
    dataset = get_dataset(out, transform=None)
    data1 = dataset.__getitem__(1)
    pprint(dataset.__getitem__(1))
    