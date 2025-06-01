from processing_dataset import *
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd, ScaleIntensityd,Resized, CastToTyped, ToTensord
from training_multitask import *
import torch
from pprint import pprint

if __name__ == "__main__":
    images_path = "/work/cuc.buithi/brats_challenge/data/train_t1_t1ce_t2_flair/imageTr"
    labels_path = "/work/cuc.buithi/brats_challenge/data/train_t1_t1ce_t2_flair/labelTr"
    cls_path = "/work/cuc.buithi/brats_challenge/code/multitask_model/data/survival_info_labeled.csv"
    
    train_transform = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    ScaleIntensityd(keys=["image"]),
    Resized(keys=["image", "label"], spatial_size=(128, 128, 128)),
    ToTensord(keys=["image", "label"]),
    CastToTyped(keys=["label"], dtype=torch.long),
    ])
    
    list_path = prepare_data_list(images_path, labels_path, cls_path)
    dataset = get_dataset(list_path, train_transform)
    dataloader = get_dataloader(dataset, batch_size=8, num_workers=2, drop_last=True, use_weighted_sampler=False)
    # pprint(list_path)
    # batch = next(iter(dataloader))
    # print("Keys:", batch.keys())
    # print("Image shape:", batch["image"].shape)
    # print("Label shape:", batch["label"].shape)
    # print("Tabular:", batch["tabular"])
    # print("Class Label:", batch["class_label"])
    # print("")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_model(dataloader, num_epochs=200, lr=1e-3, batch_size=8, alpha=0.1, resume_train=True, device=device)
    
    
