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
from monai.losses import DiceFocalLoss, DiceLoss
from monai.metrics import DiceMetric, MeanIoU
from monai.data import CacheDataset, DataLoader, Dataset
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd, ScaleIntensityd,Resized, CastToTyped, ToTensord
from multitask_model import UNETRMultitaskWithTabular


class MultitaskDataset(Dataset):
    def __init__(self, data_list, transform=None):
        self.data_list = data_list
        self.transform = transform

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx].copy()
        class_label = item.pop("class_label")
        age = item.pop("age")
        survival = item.pop("survival_days")

        if self.transform:
            item = self.transform(item)

        item["class_label"] = torch.tensor(class_label, dtype=torch.long)
        item["tabular"] = torch.tensor([age, survival], dtype=torch.float32)
        return item

def prepare_data_list(images_path, labels_path, classes_path) -> list:
    df_info = pd.read_csv(classes_path)
    df_info = df_info.dropna(subset=["Age", "Survival_days", "Extent_of_Resection_encoder"])
    
    df_info["Age"] = (df_info["Age"] - df_info["Age"].min()) / (df_info["Age"].max() - df_info["Age"].min())
    df_info["Survival_days"] = (df_info["Survival_days"] - df_info["Survival_days"].min()) / (df_info["Survival_days"].max() - df_info["Survival_days"].min())
    list_data = []
    
    for filename, (_, row) in zip(sorted(os.listdir(images_path)), df_info.iterrows()):
        image_path = os.path.join(images_path, filename)
        label_path = os.path.join(labels_path, filename)

        if not os.path.exists(image_path) or not os.path.exists(label_path):
            continue

        list_data.append({
            "image": image_path,
            "label": label_path,
            "class_label": int(row["Extent_of_Resection_encoder"]),
            "age": float(row["Age"]),
            "survival_days": float(row["Survival_days"])
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




