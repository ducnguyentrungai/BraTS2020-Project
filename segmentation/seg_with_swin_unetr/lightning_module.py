import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from metrics import Metric
from matplotlib import colors

class LitSegSwinUNETR(pl.LightningModule):
    def __init__(self, model, loss_fn, lr, optim_class=AdamW, out_path="outputs",
                 weight_decay=1e-2, num_classes=4, include_background=False):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.optim_class = optim_class
        self.weight_decay = weight_decay
        self.out_path = out_path
        self.num_classes = num_classes
        self.metric = Metric(num_classes, include_background)
        self.val_outputs = []

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch):
        x, y = batch["image"], batch["label"]

        yhat = self(x)
        loss = self.loss_fn(yhat, y) 

        metrics = {
            "iou": self.metric.IoU(yhat, y),
            "dice": self.metric.Dice(yhat, y),
            "sens": self.metric.Sensitivity(yhat, y),
            "spec": self.metric.Specificity(yhat, y),
            "acc": self.metric.Accuracy(yhat, y),
        }
        return loss, metrics


    def training_step(self, batch, batch_idx):
        loss, metrics = self.shared_step(batch)
        self.log_dict({f"train_{k}": v for k, v in metrics.items()} | {"train_loss": loss},
                      on_epoch=True, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, metrics = self.shared_step(batch)
        out = {"val_loss": loss}
        out.update({f"val_{k}": v for k, v in metrics.items()})
        self.val_outputs.append(out)

        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_dice", metrics["dice"], on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        if batch_idx == 0 and self.trainer.is_global_zero:
            self._visualize_prediction(batch, self(batch["image"]), self.current_epoch)

        return out

    def on_validation_epoch_end(self):
        if not self.val_outputs:
            return

        avg = {}
        for key in self.val_outputs[0].keys():
            values = [torch.as_tensor(o[key], device=self.device) for o in self.val_outputs]
            avg[key] = torch.stack(values).mean()

        self.log_dict(avg, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True)
        self.val_outputs.clear()


    def configure_optimizers(self):
        optimizer = self.optim_class(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = {
            "scheduler": ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10),
            "monitor": "val_loss",
            "interval": "epoch",
            "frequency": 1
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _visualize_prediction(self, batch, yhat, epoch):
        image = batch["image"][0]     # Tensor [C, H, W, D]
        label = batch["label"][0]     # Tensor [1, H, W, D] hoặc [H, W, D]
        if label.ndim == 4 and label.shape[0] == 1:
            label = label.squeeze(0)

        pred = yhat[0].argmax(dim=0)  # [H, W, D]

        mid = image.shape[-1] // 2
        img_stack = image[:, :, :, mid].cpu().numpy()  # shape: (4, H, W)
        stacked_input = [img_stack[i] for i in range(4)]  # T1, T1ce, T2, FLAIR

        gt = label[:, :, mid].cpu().numpy()
        pr = pred[:, :, mid].cpu().numpy()

        # ==== Màu segmentation ====
        seg_cmap = colors.ListedColormap(["black", "red", "green", "blue"])
        bounds = [0, 1, 2, 3]
        norm = colors.BoundaryNorm(bounds, seg_cmap.N)

        # ==== Vẽ ====
        fig, axs = plt.subplots(1, 6, figsize=(18, 4))
        titles = ["T1", "T1ce", "T2", "FLAIR", "GT", "Prediction"]

        for i in range(4):
            axs[i].imshow(stacked_input[i], cmap="gray")
            axs[i].set_title(titles[i])
            axs[i].axis("off")

        axs[4].imshow(gt, cmap=seg_cmap, norm=norm)
        axs[4].set_title("GT")
        axs[4].axis("off")

        axs[5].imshow(pr, cmap=seg_cmap, norm=norm)
        axs[5].set_title("Prediction")
        axs[5].axis("off")

        # Lưu ảnh
        save_dir = os.path.join(self.out_path, "images", f"epoch_{epoch}")
        os.makedirs(save_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "combined.png"), bbox_inches="tight")
        plt.close()

