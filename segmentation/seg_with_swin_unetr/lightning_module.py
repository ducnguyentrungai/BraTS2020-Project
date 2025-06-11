import os, csv, time
import torch
import pytorch_lightning as pl
# from monai.metrics import DiceMetric
from metrics import Metric
from tabulate import tabulate
from pytorch_lightning.callbacks import ModelCheckpoint
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.utilities.rank_zero import rank_zero_only
# class LitSegSwinUNETR(pl.LightningModule):
#     def __init__(self, model, loss_fn, lr, optim, weight_decay=1e-2, num_classes=4, include_background=False, log_dir="logs"):
#         super().__init__()
#         self.model = model
#         self.loss_fn = loss_fn
#         self.lr = lr
#         self.optim = optim
#         self.weight_decay = weight_decay
#         self.num_classes = num_classes
#         self.include_background = include_background

#         self.metric = Metric(num_classes, include_background=include_background)
#         self.log_dir = log_dir
#         os.makedirs(self.log_dir, exist_ok=True)
#         self.train_log_file = os.path.join(self.log_dir, "train_logs.csv")
#         self.val_log_file = os.path.join(self.log_dir, "val_logs.csv")

#         with open(self.train_log_file, 'w', newline='') as f:
#             csv.writer(f).writerow(['Epoch', 'Loss', 'IoU', 'Dice', 'Sensitivity', 'Specificity', 'Accuracy'])
#         with open(self.val_log_file, 'w', newline='') as f:
#             csv.writer(f).writerow(['Epoch', 'Val_Loss', 'Val_IoU', 'Val_Dice', 'Val_Sensitivity', 'Val_Specificity', 'Val_Accuracy'])

#     def forward(self, x):
#         return self.model(x)

#     def shared_step(self, batch):
#         x, y = batch["image"], batch["label"]
#         yhat = self(x)
#         loss = self.loss_fn(yhat, y)
#         iou = self.metric.IoU(yhat, y)
#         dice = self.metric.Dice(yhat, y)
#         sens = self.metric.Sensitivity(yhat, y)
#         spec = self.metric.Specificity(yhat, y)
#         acc  = self.metric.Accuracy(yhat, y)
#         return loss, iou, dice, sens, spec, acc

    
#     def training_step(self, batch, batch_idx):
#         loss, iou, dice, sens, spec, acc = self.shared_step(batch)
#         self.log_dict({
#             "train_loss": loss,
#             "train_iou": iou,
#             "train_dice": dice,
#             "train_sens": sens,
#             "train_spec": spec,
#             "train_acc": acc
#         }, on_epoch=True, prog_bar=True, sync_dist=True)

#         # if self.trainer.is_global_zero:
#         #     self.print(f"[Train][Epoch {self.current_epoch}][Batch {batch_idx}] Loss: {loss:.4f}, Dice: {dice:.4f}")
#         return loss

#     def validation_step(self, batch, batch_idx):
#         loss, iou, dice, sens, spec, acc = self.shared_step(batch)
#         self.log_dict({
#             "val_loss": loss,
#             "val_iou": iou,
#             "val_dice": dice,
#             "val_sens": sens,
#             "val_spec": spec,
#             "val_acc": acc
#         }, on_epoch=True, prog_bar=True, sync_dist=True)

#         # if self.trainer.is_global_zero:
#         #     self.print(f"[Val][Epoch {self.current_epoch}][Batch {batch_idx}] Loss: {loss:.4f}, Dice: {dice:.4f}")
#         return {"val_loss": loss, "val_iou": iou, "val_dice": dice, "val_sens": sens, "val_spec": spec, "val_acc": acc}

    
#     def on_fit_start(self):
#         if self.trainer.is_global_zero:
#             print("=" * 40)
#             print(f"[INFO] Fit started with:")
#             print(f"  Devices        : {self.trainer.num_devices}")
#             print(f"  Strategy       : {self.trainer.strategy.__class__.__name__}")
#             print(f"  World Size     : {getattr(self.trainer, 'world_size', 'unknown')}")
#             print(f"  Precision      : {self.trainer.precision_plugin.precision}")
#             print(f"  Accelerator    : {self.trainer.accelerator.__class__.__name__}")
#             print("=" * 40)

#     def on_train_epoch_end(self):
#         epoch = self.current_epoch
#         metrics = self.trainer.callback_metrics
#         self._log_to_csv(self.train_log_file, epoch, metrics, prefix='train')

#     def on_validation_epoch_end(self):
#         epoch = self.current_epoch
#         metrics = self.trainer.callback_metrics
#         self._log_to_csv(self.val_log_file, epoch, metrics, prefix='val')

#         headers = ["Set", "Loss", "IoU", "Dice", "Sensitivity", "Specificity", "Accuracy"]
#         row_train = ["Train"] + [f"{metrics.get(f'train_{k}', 0):.4f}" for k in ['loss', 'iou', 'dice', 'sens', 'spec', 'acc']]
#         row_val   = ["Val"]   + [f"{metrics.get(f'val_{k}', 0):.4f}" for k in ['loss', 'iou', 'dice', 'sens', 'spec', 'acc']]
        
#         # Hiển thị thông tin từ tất cả các rank
#         import sys
#         sys.stdout.write(f"\n[Rank {self.global_rank}] Epoch {epoch}\n")
#         sys.stdout.write(tabulate([row_train, row_val], headers=headers, tablefmt="fancy_grid") + "\n\n")
#         sys.stdout.flush()


#     def _log_to_csv(self, path, epoch, metrics, prefix='train'):
#         row = [epoch] + [metrics.get(f"{prefix}_{k}", torch.tensor(0.0)).item() for k in ['loss', 'iou', 'dice', 'sens', 'spec', 'acc']]
#         with open(path, 'a', newline='') as f:
#             csv.writer(f).writerow(row)

#     def configure_optimizers(self):
#         optimizer = self.optim(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
#         scheduler = {
#             'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10),
#             'monitor': 'val_loss',
#             'interval': 'epoch',
#             'frequency': 1
#         }
#         return {"optimizer": optimizer, "lr_scheduler": scheduler}


import torch
import pytorch_lightning as pl
from monai.metrics import DiceMetric
from metrics import Metric
from tabulate import tabulate
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.distributed as dist
from pytorch_lightning.utilities.rank_zero import rank_zero_only

def is_rank_zero():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank() == 0
    return True

def rank_zero_print(*args, **kwargs):
    if is_rank_zero():
        print(*args, **kwargs)

@rank_zero_only
def print_metrics_table(epoch, metrics):
    headers = ["Set", "Loss", "IoU", "Dice", "Sensitivity", "Specificity", "Accuracy"]
    row_train = ["Train"] + [f"{metrics.get(f'train_{k}', 0):.4f}" for k in ['loss', 'iou', 'dice', 'sens', 'spec', 'acc']]
    row_val   = ["Val"]   + [f"{metrics.get(f'val_{k}', 0):.4f}" for k in ['loss', 'iou', 'dice', 'sens', 'spec', 'acc']]

    import sys
    from tabulate import tabulate
    sys.stdout.write(f"\n[Rank 0] Epoch {epoch}\n")
    sys.stdout.write(tabulate([row_train, row_val], headers=headers, tablefmt="fancy_grid") + "\n\n")
    sys.stdout.flush()

class LitSegSwinUNETR(pl.LightningModule):
    def __init__(self, model, loss_fn, lr, optim, weight_decay=1e-2, num_classes=4, include_background=False):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.optim = optim
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.include_background = include_background

        self.metric = Metric(num_classes, include_background=include_background)

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch):
        # print(batch)
        if isinstance(batch, list) or isinstance(batch, tuple):
            batch = batch[0]  # chỉ lấy phần tử đầu tiên nếu batch là list of dicts
        x, y = batch["image"], batch["label"]
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

    def on_validation_epoch_end(self):
        epoch = self.current_epoch
        metrics = self.trainer.callback_metrics
        print_metrics_table(epoch, metrics)

    def configure_optimizers(self):
        optimizer = self.optim(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
