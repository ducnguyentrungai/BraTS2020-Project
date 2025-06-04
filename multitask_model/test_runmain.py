import os
import csv
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning import seed_everything
from tqdm import tqdm
from tabulate import tabulate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pynvml import nvmlInit, nvmlShutdown, nvmlDeviceGetCount, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo, nvmlDeviceGetUtilizationRates
from monai.losses import DiceLoss
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Spacingd, Orientationd, ScaleIntensityd, Resized, CastToTyped, ToTensord
from Metrics import Metric
from processing_dataset import prepare_data_list, get_dataset, get_dataloader
from torch.nn import SyncBatchNorm
from multitask_model import UNETRMultitaskWithTabular

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

            if mem_used_mb < threshold_mem_mb and gpu_util < threshold_util:
                gpus.append((i, mem_used_mb, gpu_util))

        if not gpus:
            return None

        gpus.sort(key=lambda x: (x[1], x[2]))
        selected_ids = [g[0] for g in gpus[:n]]
        return selected_ids

    except Exception as e:
        print(f"GPU auto-selection failed: {e}")
        return None
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
        total_loss = self.seg_alpha * loss_seg + self.cls_alpha * loss_cls
        return total_loss, loss_seg, loss_cls

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

class MultiTaskSegModule(LightningModule):
    def __init__(self, model, multitask_loss_fn, metric, lr=1e-4, num_classes=4, optimizer_fn=None):
        super().__init__()
        self.model = model
        self.loss_fn = multitask_loss_fn
        self.metric = metric
        self.lr = lr
        self.num_classes = num_classes
        self.optimizer_fn = optimizer_fn
        self.train_metrics = []
        self.val_metrics = []

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

        try:
            precision = precision_score(targets, preds, average='macro', zero_division=0)
            recall = recall_score(targets, preds, average='macro', zero_division=0)
            f1 = f1_score(targets, preds, average='macro', zero_division=0)
            cls_acc = accuracy_score(targets, preds)
        except:
            precision = recall = f1 = cls_acc = 0.0

        metrics = {
            "total_loss": total_loss.detach().cpu().item(),
            "seg_loss": seg_loss.detach().cpu().item(),
            "cls_loss": cls_loss.detach().cpu().item(),
            "dice": dice,
            "iou": iou,
            "acc": acc,
            "sensi": sensi,
            "speci": speci,
            "cls_acc": cls_acc,
            "f1": f1,
            "precision": precision,
            "recall": recall,
        }

        if stage == "train":
            self.train_metrics.append(metrics)
        else:
            self.val_metrics.append(metrics)

        self.log_dict({f"{stage}_{k}": v for k, v in metrics.items()},
                      prog_bar=(stage == 'val'), sync_dist=True, on_step=False, on_epoch=True)

        return total_loss

    def training_step(self, batch, batch_idx):
        return self.step(batch, 'train')

    def validation_step(self, batch, batch_idx):
        return self.step(batch, 'val')

    def _print_table(self):
        def avg_metrics(metrics_list):
            return {k: sum(d[k] for d in metrics_list) / len(metrics_list) for k in metrics_list[0]} if metrics_list else {}

        train_avg = avg_metrics(self.train_metrics)
        val_avg = avg_metrics(self.val_metrics)

        metric_order = [
            "total_loss", "seg_loss", "cls_loss",
            "dice", "iou", "acc", "sensi", "speci",
            "cls_acc", "f1", "precision", "recall"
        ]

        table = []
        for k in metric_order:
            train_val = train_avg.get(k, None)
            val_val = val_avg.get(k, None)
            table.append([k, f"{train_val:.4f}" if train_val is not None else "-",
                          f"{val_val:.4f}" if val_val is not None else "-"])

        print("\n\U0001F4CA Epoch Metrics Summary")
        print(tabulate(table, headers=["Metric", "Train", "Validation"], tablefmt="fancy_grid", colalign=("left", "right", "right")))

    def on_train_epoch_end(self):
        if self.trainer.sanity_checking:
            return
        self._print_table()
        self.train_metrics.clear()
        self.val_metrics.clear()

    def configure_optimizers(self):
        if callable(self.optimizer_fn):
            return self.optimizer_fn(self.parameters())
        else:
            return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=1e-6)

def train_multitask_lightning(model, multitask_loss_fn, train_loader: DataLoader, val_loader: DataLoader,
                               num_classes: int, num_epochs: int, lr: float = 1e-4,
                               include_background: bool = True, log_dir: str = "logs", num_gpus_select: int = 4,
                               optimizer_fn=None):

    selected = auto_select_gpus(n=num_gpus_select)
    if selected is None:
        selected = list(range(min(num_gpus_select, torch.cuda.device_count())))
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected))

    metric = Metric(num_classes=num_classes, include_background=include_background)
    module = MultiTaskSegModule(model=model, multitask_loss_fn=multitask_loss_fn,
                                metric=metric, lr=lr, num_classes=num_classes,
                                optimizer_fn=optimizer_fn)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_dice',
        mode='max',
        save_top_k=1,
        save_last=True,
        filename='best_model-epoch={epoch:02d}-val_dice={val_dice:.4f}',
        auto_insert_metric_name=False,
        verbose=True
    )

    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    csv_logger = CSVLogger(save_dir=log_dir, name="multitask_seg_logs")

    trainer = Trainer(
        accelerator="gpu",
        devices=len(selected),
        strategy=DDPStrategy(find_unused_parameters=True),
        sync_batchnorm=True,
        max_epochs=num_epochs,
        logger=csv_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        precision=16,
        log_every_n_steps=1
    )

    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == "__main__":
    seed_everything(42)
    images_path = "/work/cuc.buithi/brats_challenge/data/train_t1_t1ce_t2_flair/imageTr"
    labels_path = "/work/cuc.buithi/brats_challenge/data/train_t1_t1ce_t2_flair/labelTr"
    train_table_path = "/work/cuc.buithi/brats_challenge/code/multitask_model/data/train.csv"
    val_table_path = "/work/cuc.buithi/brats_challenge/code/multitask_model/data/val.csv"

    train_list = prepare_data_list(images_path, labels_path, train_table_path)
    val_list = prepare_data_list(images_path, labels_path, val_table_path)

    train_dataset = get_dataset(train_list, transform=transform_for_train())
    val_dataset = get_dataset(val_list, transform=transform_for_test())

    train_loader = get_dataloader(train_dataset, batch_size=16, num_workers=2, shuffle=True, drop_last=True)
    val_loader = get_dataloader(val_dataset, batch_size=16, num_workers=2)

    model = UNETRMultitaskWithTabular(
        in_channels=4,
        out_seg_channels=4,
        out_cls_classes=3,
        img_size=(128, 128, 128),
        tabular_dim=8,
        norm_name='batch',
        classifier_hidden_dim=[512, 256, 128, 64, 32, 16]
    )
    model = SyncBatchNorm.convert_sync_batchnorm(model)

    mul_loss = MultiTaskLoss(seg_alpha=1.0, cls_alpha=1.0)
    optimizer_fn = lambda params: torch.optim.AdamW(params, lr=2e-4, weight_decay=1e-5)

    train_multitask_lightning(
        model=model,
        multitask_loss_fn=mul_loss,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=4,
        num_epochs=300,
        lr=1e-3,
        include_background=False,
        num_gpus_select=4,
        optimizer_fn=optimizer_fn
    )
