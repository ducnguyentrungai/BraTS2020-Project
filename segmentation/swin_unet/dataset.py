import os
import glob

from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd, \
    ScaleIntensityRanged, CropForegroundd, RandCropByPosNegLabeld, RandFlipd, RandRotate90d, ToTensord
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def get_transforms(spatial_size=(128, 128, 128), is_train=True):
    base = [
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0),mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityRanged(keys=["image"], a_min=0, a_max=255,b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image", "label"], source_key="image"),
    ]
    
    if is_train:
        base += [
            RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=spatial_size,
                pos=1,
                neg=1,
                num_samples=2,
                image_key="image",
                image_threshold=0,
            ),
            RandFlipd(keys=["image", "label"], spatial_axis=[0], prob=0.2),
            RandFlipd(keys=["image", "label"], spatial_axis=[1], prob=0.2),
            RandRotate90d(keys=["image", "label"], prob=0.2, max_k=3),
        ]
    base.append(ToTensord(keys=["image", "label"]))
    return Compose(base)

def extract_data_dicts(data_dir:str):
    images = sorted(glob.glob(os.path.join(data_dir, "imageTr", "*.nii.gz")))
    labels = sorted(glob.glob(os.path.join(data_dir, "labelTr", "*.nii.gz")))
    data_dicts = [{"image": i, "label": l} for i, l in zip(images, labels)]
    return data_dicts

# def get_dataloader(data_dir, batch_size=4, is_train=True, spatial_size=(128, 128, 128), num_workers=2):
#     data_dicts = extract_data_dicts(data_dir)
#     transforms = get_transforms(spatial_size=spatial_size, is_train=is_train)

#     dataset = CacheDataset(data=data_dicts, transform=transforms, num_workers=num_workers)
#     return DataLoader(dataset, batch_size=batch_size, shuffle=is_train, num_workers=num_workers)


# if __name__ == "__main__":
#     data_dir = '/work/cuc.buithi/brats_challenge/data/train_flair_t1_t1ce_t2'
#     images = sorted(glob.glob(os.path.join(data_dir, "imageTr", "*.nii.gz")))
#     labels = sorted(glob.glob(os.path.join(data_dir, "labelTr", "*.nii.gz")))
#     data_dicts = [{"image": i, "label": l} for i, l in zip(images, labels)]
#     from pprint import pprint
#     pprint(data_dicts[0])