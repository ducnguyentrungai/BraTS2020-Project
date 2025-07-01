import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.metrics import Metric
from matplotlib import colors
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os
import matplotlib.pyplot as plt
from torch.optim import AdamW


class LitMultiTaskModule(LightningModule):
    def __init__(
        self,
        model,
        loss_fn,
        optim_class=AdamW, 
        out_path="outputs",
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        num_seg_classes: int = 4,
        include_background: bool = False,
        seg_ckpt_path: str = None
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.lr = lr
        self.weight_decay = weight_decay
        self.optim_class = optim_class
        self.out_path = out_path

        self.seg_metric = Metric(num_classes=num_seg_classes, include_background=include_background)

        # Load pretrained segmentation nếu có
        if seg_ckpt_path is not None:
            ckpt = torch.load(seg_ckpt_path, map_location="cpu")
            state_dict = ckpt["state_dict"]
            self.model.load_state_dict(state_dict, strict=False)


    def forward(self, image, tabular):
        return self.model(image, tabular)

    def shared_step(self, batch):
        image = batch["image"]
        seg_label = batch["label"]
        tabular = batch["tabular"]
        class_label = batch["label_class"]

        output = self(image, tabular)
        seg_pred = output["seg"]
        cls_pred = output["cls"]

        losses = self.loss_fn(seg_pred, seg_label, cls_pred, class_label)
        return losses, seg_pred, seg_label, cls_pred, class_label

    def training_step(self, batch, batch_idx):
        losses, _, _, _, _ = self.shared_step(batch)
        self.log("train_loss", losses["loss"], prog_bar=True, sync_dist=True)
        self.log("train_loss_seg", losses["loss_seg"], prog_bar=True, sync_dist=True)
        self.log("train_loss_cls", losses["loss_cls"], prog_bar=True, sync_dist=True)
        return losses["loss"]
        
    def validation_step(self, batch, batch_idx):
        losses, seg_pred, seg_label, cls_pred, cls_label = self.shared_step(batch)
        if batch_idx == 0:
            self._visualize_prediction(batch, seg_pred, self.current_epoch)

        # Segmentation metrics
        dice = self.seg_metric.Dice(seg_pred, seg_label)
        iou = self.seg_metric.IoU(seg_pred, seg_label)

        # Classification metrics
        y_true = cls_label.cpu().numpy()
        y_pred = cls_pred.argmax(dim=1).detach().cpu().numpy()
        print(f"Ground truth: {y_true}")
        print(f"Prediction: {y_pred}")
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        # Composite metric (bạn có thể điều chỉnh trọng số)
        composite_score = (1 - dice) * 0.8 + (1 - f1) * 0.2

        # Log all metrics
        self.log("val_total_loss", losses["loss"], sync_dist=True)
        self.log("val_loss_seg", losses["loss_seg"], sync_dist=True)
        self.log("val_loss_cls", losses["loss_cls"], sync_dist=True)
        self.log("val_dice", dice, prog_bar=True, sync_dist=True)
        self.log("val_iou", iou, sync_dist=True)
        self.log("val_cls_acc", acc, sync_dist=True)
        self.log("val_cls_prec", prec, sync_dist=True)
        self.log("val_cls_rec", rec, sync_dist=True)
        self.log("val_cls_f1", f1, prog_bar=True, sync_dist=True)
        self.log("val_composite", composite_score, prog_bar=True, sync_dist=True)

        return {
            "loss": losses["loss"],
            "dice": dice,
            "iou": iou,
            "acc": acc,
            "prec": prec,
            "rec": rec,
            "f1": f1,
            "composite": composite_score
        }

        
    def configure_optimizers(self):
        optimizer = self.optim_class(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = {
            "scheduler": ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10),
            "monitor": "val_composite",
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

