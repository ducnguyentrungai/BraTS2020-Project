import torch
import torch.nn as nn
import pytorch_lightning as pl
from monai.losses import DiceLoss
from monai.metrics import DiceMetric


class SwinUNETRModule(pl.LightningModule):
    def __init__(self, model, loss_fn, optim, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.loss_fn = loss_fn
        self.optim = optim
        self.lr = lr
        self.metric = DiceMetric(include_background=False, reduction="mean")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["label"]
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.metric(y_hat, y)
        self.log("val_loss", loss, prog_bar=True)

    def validation_epoch_end(self, outputs):
        dice = self.metric.aggregate().item()
        self.metric.reset()
        self.log("val_dice", dice, prog_bar=True)

    def configure_optimizers(self):
        return self.optim(self.parameters(), lr=self.lr)
