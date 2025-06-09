import os
import glob
from monai.data import CacheDataset, Dataset, DataLoader
from tqdm import tqdm
from pytorch_lightning import LightningDataModule

def get_transforms(spatial_size=(128, 128, 128), is_train=True):
    from monai.transforms import (
        LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd,
        ScaleIntensityRanged, CropForegroundd, Resized,
        RandCropByPosNegLabeld, RandFlipd, RandRotate90d,
        ToTensord, Compose, RandGaussianNoised, RandGaussianSmoothd
    )

    base = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
        Resized(keys=["image", "label"], spatial_size=spatial_size, mode=("trilinear", "nearest")),
    ]

    if is_train:
        base += [
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=spatial_size,
                pos=1,
                neg=1,
                num_samples=1,  # Chỉnh lại để trả về duy nhất 1 sample
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.2),
            RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.2),
            RandGaussianNoised(keys=["image"], prob=0.2),
            RandGaussianSmoothd(keys=["image"], prob=0.3),
            RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),
            
            Resized(keys=["image", "label"], spatial_size=spatial_size, mode=("trilinear", "nearest")),
        ]

    base.append(ToTensord(keys=["image", "label"]))
    return Compose(base)


def extract_data_dicts(data_dir:str, train_percent:float=0.8):
    images = sorted(glob.glob(os.path.join(data_dir, "imageTr", "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(data_dir, "labelTr", "*.nii.gz")))
    data_dicts = [{"image": i, "label": l} for i, l in zip(images, labels)]
    from sklearn.model_selection import train_test_split
    train_dicts, val_dicts = train_test_split(data_dicts, train_size=train_percent, random_state=42, shuffle=True)
    return train_dicts, val_dicts

def get_dataloader(data_dicts, batch_size=4, is_train=True, spatial_size=(128, 128, 128), num_workers=2):
    transforms = get_transforms(spatial_size=spatial_size, is_train=is_train)
    dataset = Dataset(data=data_dicts, transform=transforms)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True)



# if __name__ == "__main__":
#     data_dir = '/work/cuc.buithi/brats_challenge/data/train_flair_t1_t1ce_t2'
#     train_list, val_list = extract_data_dicts(data_dir=data_dir, train_percent=0.8)
#     print(len(train_list))
#     print(train_list[0])
    