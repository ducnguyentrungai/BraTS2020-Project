import torch
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd, ScaleIntensityd,Resized, CastToTyped, ToTensord
from monai.data import CacheDataset, DataLoader
from monai.networks.nets import UNETR
from monai.networks.nets import SwinUNETR
from monai.losses import DiceFocalLoss, DiceLoss
from monai.metrics import DiceMetric
from monai.utils import set_determinism, first
from tqdm import tqdm
import nibabel as nib
import matplotlib.pyplot as plt
from pprint import pprint
import os
import numpy as np
import torch.nn as nn
from sklearn.model_selection import train_test_split
import json
import shutil
import csv
import time
from tabulate import tabulate
from metrics import Metric


def extract_dict_data(image_paths:str, label_paths:str):
    images = [os.path.join(image_paths, image) for image in sorted(os.listdir(image_paths))]
    labels = [os.path.join(label_paths, label) for label in sorted(os.listdir(label_paths))]
    bar_prog = tqdm(zip(images, labels), desc='Procesiing')
    dictionary_data = [{'image': image, 'label': label} for image, label in bar_prog]
    return dictionary_data


def write_data(name_file: str, data: list):
    """
    Ghi dữ liệu (list hoặc dict) vào file JSON.

    Args:
        name_file (str): Tên file cần lưu (VD: 'train_data.json').
        data (list or dict): Dữ liệu cần lưu.
    """
    with open(name_file, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Đã lưu {name_file}")
    
def read_data(name_file: str):
    """
    Đọc dữ liệu từ file JSON.

    Args:
        name_file (str): Tên file cần đọc.

    Returns:
        list or dict: Dữ liệu đã đọc từ JSON.
    """
    with open(name_file, 'r') as f:
        data = json.load(f)
    print(f"Đã load {name_file}")
    return data

def transform_for_train():
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
    return transform

def transform_for_test():
    transform = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    ScaleIntensityd(keys=["image"]),
    Resized(keys=["image", "label"], spatial_size=(128, 128, 128)),
    ToTensord(keys=["image", "label"]),
    CastToTyped(keys=["label"], dtype=torch.long),
    ])
    return transform

def save_image_comparison(image, label, pred, slice_idx=69, save_path="comparison.png"):
    """
    Save original image, ground truth, and prediction slices side-by-side in a figure.

    Args:
        image (Tensor or ndarray): 3D or 5D image (B, C, D, H, W) or (D, H, W).
        label (Tensor or ndarray): Ground truth mask, shape (B, D, H, W) or (D, H, W).
        pred (Tensor or ndarray): Predicted mask, same shape as label.
        slice_idx (int): Slice index in depth dimension to visualize.
        save_path (str): Path to save the image comparison.
    """
    # Convert tensors to numpy
    if torch.is_tensor(image): image = image.detach().cpu().numpy()
    if torch.is_tensor(label): label = label.detach().cpu().numpy()
    if torch.is_tensor(pred):  pred = pred.detach().cpu().numpy()

    # Process image
    if image.ndim == 5:  # (B, C, D, H, W)
        image = image[0, 0]  # Take first batch, first channel -> (D, H, W)
    elif image.ndim == 4:  # (B, D, H, W)
        image = image[0]     # Take first batch
    elif image.ndim != 3:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    # Process label
    if label.ndim == 5:  # (B, 1, D, H, W)
        label = label[0, 0]
    elif label.ndim == 4:  # (B, D, H, W)
        label = label[0]
    elif label.ndim != 3:
        raise ValueError(f"Unsupported label shape: {label.shape}")

    # Process pred
    if pred.ndim == 5:  # (B, 1, D, H, W)
        pred = pred[0, 0]
    elif pred.ndim == 4:  # (B, D, H, W)
        pred = pred[0]
    elif pred.ndim != 3:
        raise ValueError(f"Unsupported pred shape: {pred.shape}")


    # Safety check on slice index
    if slice_idx < 0 or slice_idx >= image.shape[0]:
        raise IndexError(f"slice_idx {slice_idx} is out of range for image depth {image.shape[0]}")

    # Extract the slice
    img_slice = image[:, :, slice_idx]   # (H, W)
    label_slice = label[:, :,slice_idx]
    pred_slice = pred[:, :, slice_idx]

    # Plot and save
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(img_slice, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(label_slice, cmap='viridis')
    axes[1].set_title("Ground Truth")
    axes[1].axis('off')

    axes[2].imshow(pred_slice, cmap='viridis')
    axes[2].set_title("Prediction")
    axes[2].axis('off')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def resume_training(model, optimizer, logs_path, name_pth:str, device):
    """
    Resume training from last checkpoint saved as 'last_model.pth'.

    Returns:
        model: model with loaded weights
        optimizer: optimizer with loaded state
        start_epoch (int): epoch to continue training from
    """
    checkpoint_path = os.path.join(logs_path, name_pth)
    if not os.path.exists(checkpoint_path):
        print("Checkpoint not found, starting from scratch.")
        return model.to(device), optimizer, 0  # start from epoch 0

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1

    model.to(device)
    model.train()  # VERY IMPORTANT to continue training mode
    print(f"Resumed from epoch {start_epoch}")
    return model, optimizer, start_epoch


def training(model, loss_fn, optimization, dataloader: DataLoader, num_epochs: int, lr: float, batch_size: int, alpha: float, resume_train: bool = False, device='cpu'):
    logs_path = 'logs'
    os.makedirs(logs_path, exist_ok=True)

    file_path = os.path.join(logs_path, 'train_logs.csv')
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Loss', 'Iou', 'Dice', 'Sensitivity', 'Specificity', 'Accuracy'])

    optimizer = optimization
    metric = Metric(num_classes=4)
    num_batchs = len(dataloader)
    best_dice = 0.3
    start_time = time.time()

    if resume_train:
        model, optimizer, start_epoch = resume_training(model, optimizer, logs_path, device=device)
    else:
        start_epoch = 0

    for epoch in range(start_epoch, num_epochs):
        model.train()
        run_loss = run_iou = run_dice = run_sensi = run_speci = run_acc = 0.0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", colour="green")
        for batch in pbar:
            optimizer.zero_grad()
            image = batch['image'].to(device)
            label = batch['label'].to(device)

            output = model(image)
            loss_value = loss_fn(output, label)

            loss_value.backward()
            optimizer.step()

            run_loss += loss_value.item()
            iou = metric.IoU(output, label)
            dice = metric.Dice(output, label)
            sensi = metric.Sensitivity(output, label)
            speci = metric.Specificity(output, label)
            acc = metric.Accuracy(output, label)

            run_iou += iou
            run_dice += dice
            run_sensi += sensi
            run_speci += speci
            run_acc += acc

            pbar.set_postfix({
                "loss": f"{loss_value.item():.3f}",
                "iou": f"{iou:.3f}",
                "dice": f"{dice:.3f}",
            })

        avg_run_loss = run_loss / num_batchs
        avg_run_iou = run_iou / num_batchs
        avg_run_dice = run_dice / num_batchs
        avg_run_sensi = run_sensi / num_batchs
        avg_run_speci = run_speci / num_batchs
        avg_run_acc = run_acc / num_batchs

        with open(file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_run_loss, avg_run_iou, avg_run_dice, avg_run_sensi, avg_run_speci, avg_run_acc])

        header = ["Loss", "IoU", "Dice", "Sensitivity", "Specificity", "Accuracy"]
        values = [[f"{avg_run_loss:.4f}", f"{avg_run_iou:.4f}", f"{avg_run_dice:.4f}", f"{avg_run_sensi:.4f}", f"{avg_run_speci:.4f}", f"{avg_run_acc:.4f}"]]
        print(tabulate(values, header, tablefmt="fancy_grid"), "\n")

        if resume_train:
            if avg_run_dice > best_dice:
                best_dice = avg_run_dice
                resume_training(model, optimizer, logs_path)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()
                }, os.path.join(logs_path, 'best_model.pth'))

            # Always save the last model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(logs_path, 'last_model.pth'))

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"Training time: {int(hours)}h {int(minutes)}m {int(seconds)}s")



if __name__ == "__main__":
    image_paths = "/work/cuc.buithi/brats_challenge/data/train_t1_t1ce_t2_flair/imageTr"
    label_paths = "/work/cuc.buithi/brats_challenge/data/train_t1_t1ce_t2_flair/labelTr"
    dictionary_data = extract_dict_data(image_paths, label_paths)

    ########### TÚ VIẾT ##############

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = (128, 128, 128)  # hoặc kích thước phù hợp với dữ liệu của bạn
    lr = 0.01

    ########### TÚ VIẾT ##############

    
    # # Chia 80% train và 20% test
    # train_data, test_data = train_test_split(dictionary_data, test_size=0.2, random_state=42)
    # write_data('train_data.json', train_data)
    # write_data('test_data.json', test_data)
    
    # Đọc lại
    train_data = read_data('data_json/train_data.json')
    test_data = read_data('data_json/test_data.json')

    train_transform = transform_for_train()

    test_transform = transform_for_test()
    
    train_dataset = CacheDataset(test_data, transform=train_transform)
    train_loader = DataLoader(test_data, batch_size=8, shuffle=True, num_workers=2, drop_last=True)
    test_dataset = CacheDataset(test_data, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=8, num_workers=2)

    # # Lấy 1 mẫu trong test_dataset
    # sample = test_dataset[0]
    # print({k: v.shape for k, v in sample.items()})  # nếu sample là dict chứa tensor

    # # Hoặc lấy 1 batch từ test_loader
    # for batch in test_loader:
    #     print({k: v.shape for k, v in batch.items()})  # in shape từng tensor trong batch
    #     break  # chỉ lấy batch đầu tiên

    
    # model = UNETR(in_channels=4, 
    #             out_channels=4, 
    #             # out_cls_classes=2, 
    #             img_size=image_size, 
    #             # tabular_dim=2,
    #             norm_name='batch').to(device)

    model = UNETR(
        in_channels=4,
        out_channels=4,
        img_size=(96, 96, 96),
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        norm_name='instance',
        res_block=True
    ).to(device)


    
    loss_seg = DiceLoss(include_background=True, to_onehot_y=True, softmax=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

    ########### TÚ VIẾT ##############
    training(
        model=model,
        loss_fn=loss_seg,
        optimization=optimizer,
        dataloader=test_dataset,
        num_epochs=100,
        lr=lr,
        batch_size=8,
        alpha=0.5,  # nếu không dùng có thể bỏ qua
        resume_train=False,
        device=device
    )
    ########### TÚ VIẾT ##############

