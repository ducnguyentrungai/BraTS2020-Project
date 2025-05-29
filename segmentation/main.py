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
from evaluate import Evaluate_Model


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
    print(f"Saved {name_file} done.")
    
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
    print(f"Loaded '{name_file}' done.")
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
    def process(vol):
            if vol.ndim == 5: return vol[0, 0]
            if vol.ndim == 4: return vol[0]
            if vol.ndim == 3: return vol
            raise ValueError(f"Invalid shape: {vol.shape}")

    image = process(image)
    label = process(label)
    pred = process(pred)

    if slice_idx < 0 or slice_idx >= image.shape[0]:
        raise IndexError(f"slice_idx {slice_idx} is out of range for image depth {image.shape[0]}")

    img_slice = image[slice_idx, :, :]
    label_slice = label[slice_idx, :, :]
    pred_slice = pred[slice_idx, :, :]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    titles = ["Original Image", "Ground Truth", "Prediction"]
    for ax, slc, title in zip(axes, [img_slice, label_slice, pred_slice], titles):
        ax.imshow(slc, cmap='gray' if title == "Original Image" else 'viridis')
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()


def get_new_log_dir(base='logs'):
    os.makedirs(base, exist_ok=True)
    i = 1
    while True:
        path = os.path.join(base, f'train_{i}')
        if not os.path.exists(path):
            os.makedirs(path)
            return path
        i += 1

def resume_training(model, optimizer, checkpoint_path, device='cpu'):
    if not os.path.exists(checkpoint_path):
        print("Checkpoint not found, starting from scratch.")
        return model.to(device), optimizer, 0

    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    model.to(device)
    model.train()
    print(f"Resumed from epoch {checkpoint['epoch'] + 1}")
    return model, optimizer, checkpoint['epoch'] + 1

def training(model, loss_fn, optimizer, train_loader: DataLoader, val_loader: DataLoader,
             num_classes: int, num_epochs: int, resume_train: bool = False, device='cuda'):
    
    # === Tạo thư mục log mới giống YOLO ===
    logs_path = get_new_log_dir()
    train_log_file = os.path.join(logs_path, 'train_logs.csv')
    val_log_file = os.path.join(logs_path, 'val_logs.csv')
    best_model_path = os.path.join(logs_path, 'best_model.pth')
    last_model_path = os.path.join(logs_path, 'last_model.pth')

    # Ghi header train log
    with open(train_log_file, 'w', newline='') as f:
        csv.writer(f).writerow(['Epoch', 'Loss', 'IoU', 'Dice', 'Sensitivity', 'Specificity', 'Accuracy'])

    # Ghi header val log
    with open(val_log_file, 'w', newline='') as f:
        csv.writer(f).writerow(['Epoch', 'Val_Loss', 'Val_IoU', 'Val_Dice', 'Val_Sensitivity', 'Val_Specificity', 'Val_Accuracy'])

    metric = Metric(num_classes=num_classes)
    best_dice = 0.3
    start_time = time.time()

    model.to(device)
    if resume_train:
        model, optimizer, start_epoch = resume_training(model, optimizer, last_model_path, device=device)
    else:
        start_epoch = 0

    for epoch in range(start_epoch, num_epochs):
        model.train()
        run_loss = run_iou = run_dice = run_sensi = run_speci = run_acc = 0.0
        pbar = tqdm(train_loader, desc=f"Trainig - epoch[{epoch+1}/{num_epochs}]", colour="green")
        for batch in pbar:
            optimizer.zero_grad()
            image = batch['image'].to(device).float()
            label = batch['label'].to(device).long()

            output = model(image)
            loss_value = loss_fn(output, label)
            loss_value.backward()
            optimizer.step()

            # pred = torch.argmax(output, dim=1)
            run_loss += loss_value.item()
            run_iou += metric.IoU(output, label)
            run_dice += metric.Dice(output, label)
            run_sensi += metric.Sensitivity(output, label)
            run_speci += metric.Specificity(output, label)
            run_acc += metric.Accuracy(output, label)

            pbar.set_postfix({
                "Loss": f"{loss_value.item():.3f}",
                "Iou": f"{metric.IoU(output, label):.3f}",
                "Dice": f"{metric.Dice(output, label):.3f}",
            })

        num_batches = len(train_loader)
        avg_loss = run_loss / num_batches
        avg_iou = run_iou / num_batches
        avg_dice = run_dice / num_batches
        avg_sensi = run_sensi / num_batches
        avg_speci = run_speci / num_batches
        avg_acc = run_acc / num_batches

        # === Ghi log vào CSV ===
        with open(train_log_file, 'a', newline='') as f:
            csv.writer(f).writerow([epoch+1, avg_loss, avg_iou, avg_dice, avg_sensi, avg_speci, avg_acc])

        # === Evaluate on validation set ===
        val_loss, val_iou, val_dice, val_sensi, val_speci, val_acc = Evaluate_Model(
            model, val_loader, loss_fn, device)

        with open(val_log_file, 'a', newline='') as f:
            csv.writer(f).writerow([epoch+1, val_loss, val_iou, val_dice, val_sensi, val_speci, val_acc])

        # === In bảng gộp Train/Val (2 dòng) ===
        headers = ["Set", "Loss", "IoU", "Dice", "Sensitivity", "Specificity", "Accuracy"]
        rows = [
            ["Train", f"{avg_loss:.4f}", f"{avg_iou:.4f}", f"{avg_dice:.4f}",
             f"{avg_sensi:.4f}", f"{avg_speci:.4f}", f"{avg_acc:.4f}"],
            
            ["Val", f"{val_loss:.4f}", f"{val_iou:.4f}", f"{val_dice:.4f}",
             f"{val_sensi:.4f}", f"{val_speci:.4f}", f"{val_acc:.4f}"]
        ]
        print(tabulate(rows, headers=headers, tablefmt="fancy_grid"))

        # === Lưu mô hình tốt nhất theo Dice ===
        if val_dice > best_dice:
            best_dice = val_dice
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, best_model_path)

        # === Luôn lưu mô hình cuối ===
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
        }, last_model_path)

    elapsed = int(time.time() - start_time)
    h, m, s = elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60
    print(f"Training time: {h}h {m}m {s}s")

if __name__ == "__main__":
    
    # image_paths = "/work/cuc.buithi/brats_challenge/data/train_t1_t1ce_t2_flair/imageTr"
    # label_paths = "/work/cuc.buithi/brats_challenge/data/train_t1_t1ce_t2_flair/labelTr"
    # dictionary_data = extract_dict_data(image_paths, label_paths)
    
    # # Chia 80% train và 20% test
    # train_data, test_data = train_test_split(dictionary_data, test_size=0.2, random_state=42)
    # write_data('train_data.json', train_data)
    # write_data('test_data.json', test_data)
    
     
    # Đọc lại
    train_data = read_data('data_json/train_data.json')
    test_data = read_data('data_json/test_data.json')

    train_transform = transform_for_train()
    test_transform = transform_for_test()
    
    train_dataset = CacheDataset(train_data, transform=train_transform)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=2, drop_last=True)
    test_dataset = CacheDataset(test_data, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=8, num_workers=2)


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = UNETR(
        in_channels=4,
        out_channels=4,
        img_size=(128, 128, 128), 
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        norm_name='batch',
        res_block=True
    ).to(device)

    loss_seg = DiceLoss(include_background=True, to_onehot_y=True, softmax=True)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-6)

    training(
        model=model,
        loss_fn=loss_seg,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=test_loader,
        num_classes=4,
        num_epochs=300,
        resume_train=False,
        device=device
    )


