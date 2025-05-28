from tqdm import tqdm
import tabulate
import torch
import torch.nn as nn
import torch.optim as optim
from monai.utils import first
from monai.losses import DiceFocalLoss, DiceLoss
from monai.metrics import DiceMetric, MeanIoU
from multitask_model import UNETRMultitaskWithTabular
from Metric import Metric
import wandb
import os
import time
import csv
from tabulate import tabulate
import matplotlib.pyplot as plt
import copy
import shutil


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

class UncertaintyWeightedLoss(nn.Module):
    def __init__(self):
        super(UncertaintyWeightedLoss, self).__init__()
        # Gồm 2 nhiệm vụ: segmentation và classification
        self.log_var_seg = nn.Parameter(torch.tensor(0.0))  # log(σ_seg^2)
        self.log_var_cls = nn.Parameter(torch.tensor(0.0))  # log(σ_cls^2)

    def forward(self, loss_seg, loss_cls):
        # precision = 1 / σ^2 = exp(-log_var)
        precision_seg = torch.exp(-self.log_var_seg)
        precision_cls = torch.exp(-self.log_var_cls)

        total_loss = (
            precision_seg * loss_seg + self.log_var_seg +
            precision_cls * loss_cls + self.log_var_cls
        )
        return total_loss

def resume_training(model, optimizer, logs_path, device):
    """
    Resume training from last checkpoint saved as 'last_model.pth'.

    Returns:
        model: model with loaded weights
        optimizer: optimizer with loaded state
        start_epoch (int): epoch to continue training from
    """
    checkpoint_path = os.path.join(logs_path, 'last_model.pth')
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


def train_model(dataloader, num_epochs:int, lr:float, batch_size:int, alpha:float, resume_train:bool=False, device='cpu'):     
    logs_path = 'logs'
    out_pred_path = '/work/cuc.buithi/brats_challenge/code/predict_images'
    if os.path.isdir(out_pred_path):
        shutil.rmtree(out_pred_path)
    else:
        os.makedirs(out_pred_path)
    
    
    os.makedirs(logs_path, exist_ok=True)
    file_path = os.path.join(logs_path, 'train_logs.csv')
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Total Loss', 'Segmen Loss', 'Class Loss', 'Iou', 'Dice', 'Sensitivity', 'Specificity', 'Segmen Accuracy', 'Class Accuracy'])
        
    model = UNETRMultitaskWithTabular(in_channels=4, 
                                      out_seg_channels=4, 
                                      out_cls_classes=2, 
                                      img_size=128, 
                                      tabular_dim=2,
                                      norm_name='batch').to(device)
    
    loss_seg = DiceLoss(include_background=True, to_onehot_y=True, softmax=True)
    loss_cls = nn.CrossEntropyLoss()
    # uw_loss = UncertaintyWeightedLoss()
    # optimizer = torch.optim.Adam([{'params': model.parameters()},
    #                               {'params': uw_loss.parameters()}], 
    #                              lr=1e-3, weight_decay=1e-6)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)
    metric = Metric(num_classes=4)
    num_batchs = len(dataloader)
    start_time = time.time()
    best_dice = 0.3
    count = 0
    epoch = 0
    
    if resume_train:
        model, optimizer, epoch = resume_training(model, optimizer, logs_path, device=device)

    for epoch in range(num_epochs):
        model.train()
        run_total_loss = run_seg_loss = run_cls_loss = 0.0
        run_iou = run_dice = run_sensi = run_speci = run_acc = total_cls = 0.0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", colour="green")
        for batch in pbar:
            optimizer.zero_grad()
            
            image = batch['image'].to(device)
            label = batch['label'].to(device)
            tabular = batch['tabular'].to(device)
            class_label = batch['class_label'].to(device)
            
            seg_output, cls_output = model(image, tabular)
            pred = torch.argmax(seg_output, dim=1)

            save_image_comparison(image=image, label=label, pred=pred, slice_idx=69, save_path=os.path.join(out_pred_path,f'out_{count}.png'))
            count += 1

            l_seg = loss_seg(seg_output, label)
            l_cls = loss_cls(cls_output, class_label)
            # loss = uw_loss(l_seg, l_cls)
            loss = (1.0 - alpha)*l_seg + alpha* l_cls + 1e-8
            run_seg_loss += l_seg.item()
            run_cls_loss += l_cls.item()
            run_total_loss += loss.item()
    
            loss.backward()
            optimizer.step()
            
            iou = metric.IoU(seg_output, label)
            dice = metric.Dice(seg_output, label)
            sensi = metric.Sensitivity(seg_output, label)
            speci = metric.Specificity(seg_output, label)
            acc = metric.Accuracy(seg_output, label)
            
            run_iou += iou
            run_dice += dice
            run_sensi += sensi
            run_speci += speci
            run_acc += acc
            
            # Classification Accuracy
            cls_pred = torch.argmax(cls_output, dim=1)
            cls_acc = (cls_pred == class_label).sum().item() / class_label.numel()
            total_cls += cls_acc

            pbar.set_postfix({"total_loss": f"{loss.item():.3f}",
                    "seg_loss": f"{l_seg.item():.3f}",
                    "cls_loss": f"{l_cls.item():.3f}",
                    # "seg_dice" : f"{dice:.4f}",
                    # "cls_acc": f"{cls_acc:.4f}"
                    })  

        avg_run_total_loss = f"{run_total_loss / num_batchs:.4f}"
        avg_run_seg_loss = f"{run_seg_loss / num_batchs:.4f}"
        avg_run_cls_loss = f"{run_cls_loss / num_batchs:.4f}"
        avg_run_iou = f"{run_iou / num_batchs:.4f}"
        avg_run_dice = f"{run_dice / num_batchs:.4f}"
        temp = run_dice / num_batchs
        avg_run_sensi = f"{run_sensi / num_batchs:.4f}"
        avg_run_speci = f"{run_speci / num_batchs:.4f}"
        avg_seg_acc = f"{run_acc / num_batchs:.4f}"
        avg_cls_acc =  f"{total_cls / num_batchs:.4f}"
        with open(file_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_run_total_loss, avg_run_seg_loss, avg_run_cls_loss, avg_run_iou, avg_run_dice, avg_run_sensi, avg_run_speci, avg_seg_acc, avg_cls_acc])
        
        header = ["m_total_loss", "m_seg_loss", "m_cls_loss", "mIoU", "mDice", "mSensitivity", "mSpecificity", "m_acc_seg", "m_acc_cls"]
        values = [[avg_run_total_loss, avg_run_seg_loss, avg_run_cls_loss, avg_run_iou, avg_run_dice, avg_run_sensi, avg_run_speci, avg_seg_acc,avg_cls_acc]]
        print(tabulate(values, header, tablefmt="fancy_grid"), "\n")
        
        if temp > best_dice:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict()
            }, os.path.join(logs_path, 'best_model.pth'))
            temp = best_dice
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
       