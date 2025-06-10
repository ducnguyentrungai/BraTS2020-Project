import os, csv, time
import torch
import pytorch_lightning as pl
from monai.metrics import DiceMetric
from metrics import Metric
from tabulate import tabulate
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm

class LitSegSwinUNETR(pl.LightningModule):
    def __init__(self, model, loss_fn, optim, lr, num_classes=4, include_background=False, log_dir="logs"):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optim = optim
        self.lr = lr
        self.num_classes = num_classes
        self.include_background = include_background

        self.metric = Metric(num_classes, include_background=include_background)
        self.log_dir = log_dir
        os.makedirs(self.log_dir, exist_ok=True)
        self.train_log_file = os.path.join(self.log_dir, "train_logs.csv")
        self.val_log_file = os.path.join(self.log_dir, "val_logs.csv")

        with open(self.train_log_file, 'w', newline='') as f:
            csv.writer(f).writerow(['Epoch', 'Loss', 'IoU', 'Dice', 'Sensitivity', 'Specificity', 'Accuracy'])
        with open(self.val_log_file, 'w', newline='') as f:
            csv.writer(f).writerow(['Epoch', 'Val_Loss', 'Val_IoU', 'Val_Dice', 'Val_Sensitivity', 'Val_Specificity', 'Val_Accuracy'])

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch):
        x, y = batch["image"], batch["label"]
        print(f"[RANK {self.global_rank}] Batch device: {x.device}")
        yhat = self(x)
        loss = self.loss_fn(yhat, y)
        iou = self.metric.IoU(yhat, y)
        dice = self.metric.Dice(yhat, y)
        sens = self.metric.Sensitivity(yhat, y)
        spec = self.metric.Specificity(yhat, y)
        acc  = self.metric.Accuracy(yhat, y)
        return loss, iou, dice, sens, spec, acc

    
    def training_step(self, batch, batch_idx):
        loss, iou, dice, sens, spec, acc = self.shared_step(batch)
        self.log_dict({
            "train_loss": loss,
            "train_iou": iou,
            "train_dice": dice,
            "train_sens": sens,
            "train_spec": spec,
            "train_acc": acc
        }, on_epoch=True, prog_bar=True, sync_dist=True)

        if self.trainer.is_global_zero:
            self.print(f"[Train][Epoch {self.current_epoch}][Batch {batch_idx}] Loss: {loss:.4f}, Dice: {dice:.4f}")
        return loss

    def validation_step(self, batch, batch_idx):
        loss, iou, dice, sens, spec, acc = self.shared_step(batch)
        self.log_dict({
            "val_loss": loss,
            "val_iou": iou,
            "val_dice": dice,
            "val_sens": sens,
            "val_spec": spec,
            "val_acc": acc
        }, on_epoch=True, prog_bar=True, sync_dist=True)

        if self.trainer.is_global_zero:
            self.print(f"[Val][Epoch {self.current_epoch}][Batch {batch_idx}] Loss: {loss:.4f}, Dice: {dice:.4f}")
        return {"val_loss": loss, "val_iou": iou, "val_dice": dice, "val_sens": sens, "val_spec": spec, "val_acc": acc}

    
    def on_fit_start(self):
        if self.trainer.is_global_zero:
            print("=" * 40)
            print(f"[INFO] Fit started with:")
            print(f"  Devices        : {self.trainer.num_devices}")
            print(f"  Strategy       : {self.trainer.strategy.__class__.__name__}")
            print(f"  World Size     : {getattr(self.trainer, 'world_size', 'unknown')}")
            print(f"  Precision      : {self.trainer.precision_plugin.precision}")
            print(f"  Accelerator    : {self.trainer.accelerator.__class__.__name__}")
            print("=" * 40)

    def on_train_epoch_end(self):
        epoch = self.current_epoch
        metrics = self.trainer.callback_metrics
        self._log_to_csv(self.train_log_file, epoch, metrics, prefix='train')

    def on_validation_epoch_end(self):
        epoch = self.current_epoch
        metrics = self.trainer.callback_metrics
        self._log_to_csv(self.val_log_file, epoch, metrics, prefix='val')

        headers = ["Set", "Loss", "IoU", "Dice", "Sensitivity", "Specificity", "Accuracy"]
        row_train = ["Train"] + [f"{metrics.get(f'train_{k}', 0):.4f}" for k in ['loss', 'iou', 'dice', 'sens', 'spec', 'acc']]
        row_val   = ["Val"]   + [f"{metrics.get(f'val_{k}', 0):.4f}" for k in ['loss', 'iou', 'dice', 'sens', 'spec', 'acc']]
        print(tabulate([row_train, row_val], headers=headers, tablefmt="fancy_grid"), '\n')

    def _log_to_csv(self, path, epoch, metrics, prefix='train'):
        row = [epoch] + [metrics.get(f"{prefix}_{k}", torch.tensor(0.0)).item() for k in ['loss', 'iou', 'dice', 'sens', 'spec', 'acc']]
        with open(path, 'a', newline='') as f:
            csv.writer(f).writerow(row)

    def configure_optimizers(self):
        return self.optim(self.parameters(), lr=self.lr)
