import os
import csv
import time
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies import DDPStrategy
from tqdm import tqdm
from tabulate import tabulate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report, log_loss
from pynvml import nvmlInit, nvmlShutdown, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates
from Metrics import Metric
from processing_dataset import *


def auto_select_gpus(n=3, threshold_mem_mb=800, threshold_util=10):
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
            return None

        gpus.sort(key=lambda x: (x[1], x[2]))
        selected_ids = [g[0] for g in gpus[:n]]
        print(f"✅ Auto-selected GPUs: {selected_ids}")
        return selected_ids

        gpus.sort(key=lambda x: (x[1], x[2]))
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

class MultiTaskLoss(nn.Module):
    def __init__(self, seg_alpha=1.0, cls_alpha=1.0,
                 include_background=True, to_onehot_y=True, softmax=True):
        super(MultiTaskLoss, self).__init__()
        self.seg_loss_fn = DiceLoss(
            include_background=include_background,
            to_onehot_y=to_onehot_y,
            softmax=softmax
        )
        self.cls_loss_fn = nn.CrossEntropyLoss()
        self.seg_alpha = seg_alpha
        self.cls_alpha = cls_alpha

    def forward(self, seg_output, seg_target, cls_output, cls_target):
        loss_seg = self.seg_loss_fn(seg_output, seg_target)
        loss_cls = self.cls_loss_fn(cls_output, cls_target)
        total_loss = self.seg_alpha * loss_seg + self.cls_alpha * loss_cls + 1e-8
        return total_loss, loss_seg, loss_cls 

class MultiTaskSegModule(LightningModule):
    def __init__(self, model, multitask_loss_fn, metric, lr=1e-4, num_classes=4, include_background=True):
        super().__init__()
        self.model = model
        self.loss_fn = multitask_loss_fn
        self.metric = metric
        self.lr = lr
        self.num_classes = num_classes
        self.include_background = include_background

    def forward(self, x_img, x_tabular):
        return self.model(x_img, x_tabular)

    def step(self, batch, stage):
        image = batch['image']
        label = batch['label']
        tabular = batch['tabular']
        class_label = batch['class_label']

        seg_output, cls_output = self.model(image, tabular)
        total_loss, seg_loss, cls_loss = self.loss_fn(seg_output, label, cls_output, class_label)

        iou = self.metric.IoU(seg_output, label)
        dice = self.metric.Dice(seg_output, label)
        sensi = self.metric.Sensitivity(seg_output, label)
        speci = self.metric.Specificity(seg_output, label)
        acc = self.metric.Accuracy(seg_output, label)

        preds = torch.argmax(cls_output, dim=1).cpu().numpy()
        targets = class_label.cpu().numpy()

        precision = precision_score(targets, preds, average='macro', zero_division=0)
        recall = recall_score(targets, preds, average='macro', zero_division=0)
        f1 = f1_score(targets, preds, average='macro', zero_division=0)
        cls_acc = accuracy_score(targets, preds)

        self.log_dict({
            f'{stage}_total_loss': total_loss,
            f'{stage}_seg_loss': seg_loss,
            f'{stage}_cls_loss': cls_loss,
            f'{stage}_iou': iou,
            f'{stage}_dice': dice,
            f'{stage}_sensi': sensi,
            f'{stage}_speci': speci,
            f'{stage}_acc': acc,
            f'{stage}_precision': precision,
            f'{stage}_recall': recall,
            f'{stage}_f1': f1,
            f'{stage}_cls_acc': cls_acc,
        }, prog_bar=(stage == 'val'), sync_dist=True, on_step=False, on_epoch=True)

        return total_loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, 'val')

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


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

def train_multitask_lightning(model, multitask_loss_fn, train_loader: DataLoader, val_loader: DataLoader,
                               num_classes: int, num_epochs: int, lr: float = 1e-4,
                               include_background: bool = True, log_dir: str = "logs", num_gpus_select: int = 3):

    selected = auto_select_gpus(n=num_gpus_select)
    if selected is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected))

      # Đảm bảo file metric.py có Metric
    metric = Metric(num_classes=num_classes, include_background=include_background)

    module = MultiTaskSegModule(model, multitask_loss_fn, metric, lr=lr,
                                num_classes=num_classes, include_background=include_background)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_dice',
        mode='max',
        save_top_k=1,
        filename='best_model-{epoch:02d}-{val_dice:.4f}',
        verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    csv_logger = CSVLogger(save_dir=log_dir, name="multitask_seg_logs")

    num_gpus = torch.cuda.device_count()

    trainer = Trainer(
        accelerator="gpu" if num_gpus > 0 else "cpu",
        devices=num_gpus if num_gpus > 0 else None,
        strategy=DDPStrategy(find_unused_parameters=True) if num_gpus > 1 else None,
        max_epochs=num_epochs,
        logger=csv_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        precision=16 if torch.cuda.is_available() else 32,
        log_every_n_steps=5
    )

    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)


if __name__ == "__main__":
    images_path = "/work/cuc.buithi/brats_challenge/data/train_t1_t1ce_t2_flair/imageTr"
    labels_path = "/work/cuc.buithi/brats_challenge/data/train_t1_t1ce_t2_flair/labelTr"
    train_table_path = "/work/cuc.buithi/brats_challenge/code/multitask_model/data/train.csv"
    val_table_path = "/work/cuc.buithi/brats_challenge/code/multitask_model/data/val.csv"
    train_list = prepare_data_list(images_path, labels_path, train_table_path)
    val_list = prepare_data_list(images_path, labels_path, val_table_path)
    
    train_dataset = get_dataset(train_list, transform=transform_for_train())
    val_dataset = get_dataset(val_list, transform=transform_for_test())
    
    train_loader = get_dataloader(train_dataset, batch_size=8, num_workers=2, shuffle=True, drop_last=True)
    val_loader = get_dataloader(val_dataset, batch_size=8, num_workers=2)
    
     # === Model ===
    model = UNETRMultitaskWithTabular(
        in_channels=4,
        out_seg_channels=3,
        out_cls_classes=2,
        img_size=(128, 128, 128),
        tabular_dim=8,
        classifier_hidden_dim=[256, 128, 64]
    )

    # === Train ===
    mul_loss = MultiTaskLoss(seg_alpha=1.0, cls_alpha=1.0)
    train_multitask_lightning(
        model=model,
        multitask_loss_fn=mul_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=4,
        num_epochs=100,
        lr=1e-4,
        include_background=False,
        num_gpus_select=4
    )
