# ==== IMPORTS ====
import os
import json
import csv
import time
import shutil
import torch
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd, ScaleIntensityd, Resized, ToTensord, CastToTyped
from monai.data import Dataset, DataLoader
from monai.networks.nets import SwinUNETR
from monai.losses import DiceLoss

from metrics import Metric
from evaluate import Evaluate_Model
from pynvml import *

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies import DDPStrategy

class Metric:
    def __init__(self, num_classes: int, include_background: bool = True):
        self.num_classes = num_classes
        self.include_background = include_background

    def _prepare_preds(self, preds: torch.Tensor) -> torch.Tensor:
        if preds.ndim == 5:
            return torch.argmax(preds, dim=1)  # [B, D, H, W]
        elif preds.ndim == 4:
            return preds
        else:
            raise ValueError(f"Expected 4D or 5D preds, got shape {preds.shape}")

    def _prepare_targets(self, targets: torch.Tensor) -> torch.Tensor:
        return targets

    def _confusion_elements(self, preds, targets, cls):
        tp = ((preds == cls) & (targets == cls)).sum().float()
        fp = ((preds == cls) & (targets != cls)).sum().float()
        fn = ((preds != cls) & (targets == cls)).sum().float()
        tn = ((preds != cls) & (targets != cls)).sum().float()
        return tp, fp, fn, tn

    def Dice(self, preds, targets):
        preds = self._prepare_preds(preds).to(targets.device)
        targets = self._prepare_targets(targets).to(targets.device)
        smooth = 1e-5
        dices = []
        for cls in range(int(self.include_background), self.num_classes):
            tp, _, fn, _ = self._confusion_elements(preds, targets, cls)
            dice = (2 * tp + smooth) / (2 * tp + fn + fn + smooth)
            dices.append(dice)
        return torch.mean(torch.stack(dices))

    def IoU(self, preds, targets):
        preds = self._prepare_preds(preds).to(targets.device)
        targets = self._prepare_targets(targets).to(targets.device)
        smooth = 1e-5
        ious = []
        for cls in range(int(self.include_background), self.num_classes):
            tp, fp, fn, _ = self._confusion_elements(preds, targets, cls)
            iou = (tp + smooth) / (tp + fp + fn + smooth)
            ious.append(iou)
        return torch.mean(torch.stack(ious))

    def Sensitivity(self, preds, targets):
        preds = self._prepare_preds(preds).to(targets.device)
        targets = self._prepare_targets(targets).to(targets.device)
        sensitivities = []
        smooth = 1e-5
        for cls in range(int(self.include_background), self.num_classes):
            tp, _, fn, _ = self._confusion_elements(preds, targets, cls)
            sensitivity = (tp + smooth) / (tp + fn + smooth)
            sensitivities.append(sensitivity)
        return torch.mean(torch.stack(sensitivities))

    def Specificity(self, preds, targets):
        preds = self._prepare_preds(preds).to(targets.device)
        targets = self._prepare_targets(targets).to(targets.device)
        specificities = []
        smooth = 1e-5
        for cls in range(int(self.include_background), self.num_classes):
            _, fp, _, tn = self._confusion_elements(preds, targets, cls)
            specificity = (tn + smooth) / (tn + fp + smooth)
            specificities.append(specificity)
        return torch.mean(torch.stack(specificities))

    def Accuracy(self, preds, targets):
        preds = self._prepare_preds(preds).to(targets.device)
        targets = self._prepare_targets(targets).to(targets.device)
        correct = (preds == targets).sum().float()
        total = torch.numel(targets)
        return correct / (total + 1e-5)


# ==== GPU AUTO-SELECTION ====
def auto_select_gpus(n=2, threshold_mem_mb=1000, threshold_util=10):
    """
    Trả về n GPU rảnh nhất (RAM & load thấp), dùng cho DDP hoặc multi-GPU.
    """
    try:
        nvmlInit()
        device_count = nvmlDeviceGetCount()
        gpus = []

        for i in range(device_count):
            handle = nvmlDeviceGetHandleByIndex(i)
            mem_info = nvmlDeviceGetMemoryInfo(handle)
            util_info = nvmlDeviceGetUtilizationRates(handle)

            mem_used_mb = mem_info.used // 1024**2
            gpu_util = util_info.gpu

            print(f"GPU {i}: Used {mem_used_mb}MB, Util {gpu_util}%")

            if mem_used_mb < threshold_mem_mb and gpu_util < threshold_util:
                gpus.append((i, mem_used_mb, gpu_util))

        if not gpus:
            print("⚠️ No suitable GPU found. Using default CUDA_VISIBLE_DEVICES.")
            return

        # Sort and pick top n
        gpus.sort(key=lambda x: (x[1], x[2]))  # sort by (mem, util)
        selected_ids = [str(g[0]) for g in gpus[:n]]
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(selected_ids)
        print(f"✅ Auto-selected GPUs: {selected_ids}")

    except Exception as e:
        print(f"⚠️ GPU auto-selection failed: {e}")
    finally:
        try:
            nvmlShutdown()
        except:
            pass

# ==== UTILITY FUNCTIONS ====
def read_data(name_file: str):
    with open(name_file, 'r') as f:
        data = json.load(f)
    print(f"Loaded '{name_file}' done.")
    return data

def transform_for_train():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")),
        Orientationd(keys=["image", "label"], axcodes="RAS"),
        ScaleIntensityd(keys=["image"]),
        Resized(keys=["image", "label"], spatial_size=(128, 128, 128)),
        ToTensord(keys=["image", "label"]),
        CastToTyped(keys=["label"], dtype=torch.long),
    ])

def transform_for_test():
    return Compose([
        LoadImaged(keys=["image", "label"]),
        EnsureChannelFirstd(keys=["image", "label"]),
        ScaleIntensityd(keys=["image"]),
        Resized(keys=["image", "label"], spatial_size=(128, 128, 128)),
        ToTensord(keys=["image", "label"]),
        CastToTyped(keys=["label"], dtype=torch.long),
    ])


# ==== LIGHTNING MODULE ====
class SegModule(pl.LightningModule):
    def __init__(self, model, loss_fn, metric, lr=1e-4, num_classes=4, include_background=True):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.metric = metric
        self.lr = lr
        self.num_classes = num_classes
        self.include_background = include_background

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        image = batch['image'].to(self.device)
        label = batch['label'].to(self.device)
        output = self.model(image)
        loss = self.loss_fn(output, label)

        iou = self.metric.IoU(output, label)
        dice = self.metric.Dice(output, label)
        sensi = self.metric.Sensitivity(output, label)
        speci = self.metric.Specificity(output, label)
        acc = self.metric.Accuracy(output, label)

        self.log_dict({'train_loss': loss, 'train_iou': iou, 'train_dice': dice,
                       'train_sensi': sensi, 'train_speci': speci, 'train_acc': acc}, 
                      prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        image = batch['image'].to(self.device)
        label = batch['label'].to(self.device)
        output = self.model(image)
        loss = self.loss_fn(output, label)

        iou = self.metric.IoU(output, label)
        dice = self.metric.Dice(output, label)
        sensi = self.metric.Sensitivity(output, label)
        speci = self.metric.Specificity(output, label)
        acc = self.metric.Accuracy(output, label)

        self.log_dict({'val_loss': loss, 'val_iou': iou, 'val_dice': dice,
                       'val_sensi': sensi, 'val_speci': speci, 'val_acc': acc}, 
                      prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


# ==== TRAINING FUNCTION ====
def train_with_lightning(model, train_loader, val_loader, loss_fn, num_classes=4, max_epochs=100, lr=1e-4):
    # Tự động chọn GPU rảnh nhất trước khi khởi tạo Trainer
    auto_select_gpus(n=4, threshold_mem_mb=2000, threshold_util=15)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if visible:
        devices = list(map(int, visible.split(",")))
        accelerator = "gpu"
    else:
        devices = 1
        accelerator = "cpu"
    seed_everything(42)
    metric = Metric(num_classes=num_classes, include_background=False)
    module = SegModule(model, loss_fn, metric, lr=lr, num_classes=num_classes)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_dice',
        mode='max',
        save_top_k=1,
        filename='best_model-{epoch:02d}-{val_dice:.4f}',
        verbose=True
    )

    csv_logger = CSVLogger(save_dir="logs", name="segmentation_logs")
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    import os
    gpus = torch.cuda.device_count()
    if gpus == 0:
        devices = 1
        accelerator = "cpu"
    else:
        devices = list(range(min(gpus, 4)))
        accelerator = "gpu"

    trainer = Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=DDPStrategy(find_unused_parameters=True),
        max_epochs=max_epochs,
        logger=csv_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        precision=16,
        log_every_n_steps=5,
    )

    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)


# ==== MAIN SCRIPT ====
if __name__ == "__main__":
    train_data = read_data('/work/cuc.buithi/brats_challenge/code/segmentation/data_json/train_data.json')
    test_data = read_data('/work/cuc.buithi/brats_challenge/code/segmentation/data_json/test_data.json')

    train_transform = transform_for_train()
    test_transform = transform_for_test()

    train_dataset = Dataset(train_data, transform=train_transform)
    val_dataset = Dataset(test_data, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=2, num_workers=2)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model = SwinUNETR(
        img_size=(128, 128, 128),
        in_channels=4,
        out_channels=4,
        depths=(2, 2, 2, 2),
        num_heads=(3, 6, 12, 24),
        feature_size=24,
        norm_name='batch',
        drop_rate=0.0,
        attn_drop_rate=0.0,
        dropout_path_rate=0.0,
        normalize=True,
        use_checkpoint=False,
        spatial_dims=3,
        downsample='merging',
        use_v2=False
    ).to(device)

    loss_fn = DiceLoss(
        include_background=True,
        to_onehot_y=True,
        softmax=True,
    )

    train_with_lightning(model, train_loader, val_loader, loss_fn, num_classes=4, max_epochs=100)
