import pytorch_lightning as pl
from metrics import Metric
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.distributed as dist
from pytorch_lightning.utilities.rank_zero import rank_zero_only
import os
import pandas as pd
import torch
import matplotlib.pyplot as plt

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

@rank_zero_only
def save_metrics_csv(epoch, metrics, path="logs/swin_unetr_metrics.csv"):
    row = {"epoch": epoch}
    keys = ['train_loss', 'train_iou', 'train_dice', 'train_sens', 'train_spec', 'train_acc',
            'val_loss', 'val_iou', 'val_dice', 'val_sens', 'val_spec', 'val_acc']
    
    for k in keys:
        v = metrics.get(k)
        if isinstance(v, torch.Tensor):
            v = v.item()
        row[k] = v if v is not None else float("nan")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        df = pd.read_csv(path)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    except FileNotFoundError:
        df = pd.DataFrame([row])
    df.to_csv(path, index=False)

def show_image(image, title=""):
    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap="gray")
    plt.title(title)
    plt.axis("off")
    plt.show()

class LitSegSwinUNETR(pl.LightningModule):
    def __init__(self, model, loss_fn, lr, optim, out_path:str, weight_decay=1e-2, num_classes=4, include_background=False):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.optim = optim
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.include_background = include_background
        self.out_path = out_path

        self.metric = Metric(num_classes, include_background=include_background)

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch):
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
        }, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
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
        }, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        # 🧪 Visualization
        if batch_idx == 0 and self.trainer.is_global_zero:  # chỉ in 1 batch đầu
            x = batch["image"][0]  # (4, 128, 128, 128)
            y = batch["label"][0]  # (128, 128, 128)
            yhat = self(batch["image"])[0].argmax(dim=0)  # dự đoán nhãn, shape: (128, 128, 128)

            mid_slice = x.shape[2] // 2
            input_img = x[0, :, :, mid_slice].detach().cpu().numpy()
            gt_mask = y[:, :, mid_slice].detach().cpu().numpy()
            pred_mask = yhat[:, :, mid_slice].detach().cpu().numpy()

            show_image(input_img, "Input image (modal 0, mid slice)")
            show_image(gt_mask, "Ground Truth")
            show_image(pred_mask, "Predicted")
            
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
        save_metrics_csv(epoch, metrics, path=self.out_path)
        
    def configure_optimizers(self):
        optimizer = self.optim(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = {
            'scheduler': ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
