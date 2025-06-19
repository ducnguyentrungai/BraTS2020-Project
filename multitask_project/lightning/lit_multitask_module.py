import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.metrics import Metric


class LitMultiTaskModule(LightningModule):
    def __init__(
        self,
        model,
        loss_fn,
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

        self.seg_metric = Metric(num_classes=num_seg_classes, include_background=include_background)

        # Load pretrained segmentation nếu có
        if seg_ckpt_path is not None:
            self.model.load_pretrained_segmentation(seg_ckpt_path)

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
        self.log("train_loss_seg", losses["loss_seg"], sync_dist=True)
        self.log("train_loss_cls", losses["loss_cls"], sync_dist=True)
        return losses["loss"]

    def validation_step(self, batch, batch_idx):
        losses, seg_pred, seg_label, cls_pred, cls_label = self.shared_step(batch)

        # Segmentation metrics
        dice = self.seg_metric.Dice(seg_pred, seg_label)
        iou = self.seg_metric.IoU(seg_pred, seg_label)

        # Classification metrics
        y_true = cls_label.cpu().numpy()
        y_pred = cls_pred.argmax(dim=1).cpu().numpy()
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

        self.log("val_loss", losses["loss"], prog_bar=True, sync_dist=True)
        self.log("val_loss_seg", losses["loss_seg"], sync_dist=True)
        self.log("val_loss_cls", losses["loss_cls"], sync_dist=True)
        self.log("val_dice", dice, prog_bar=True, sync_dist=True)
        self.log("val_iou", iou, sync_dist=True)
        self.log("val_cls_acc", acc, sync_dist=True)
        self.log("val_cls_prec", prec, sync_dist=True)
        self.log("val_cls_rec", rec, sync_dist=True)
        self.log("val_cls_f1", f1, sync_dist=True)

        return {
            "loss": losses["loss"],
            "dice": dice,
            "iou": iou,
            "acc": acc,
            "prec": prec,
            "rec": rec,
            "f1": f1
        }

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
