import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics import Accuracy, AUROC, F1


def focal_loss(preds, targets, alpha=0.25, gamma=1.5):
    bce_loss = F.binary_cross_entropy_with_logits(
        preds, targets.to(preds.dtype), reduction="none"
    )
    probs = torch.sigmoid(preds)
    loss = torch.where(
        targets >= 0.5,
        alpha * (1.0 - probs) ** gamma * bce_loss,
        (1 - alpha) * probs ** gamma * bce_loss,
    )
    loss = loss.mean()
    return loss


class HPAClassifier(pl.LightningModule):
    def __init__(
        self,
        model_name="tf_efficientnet_b0",
        pretrained=True,
        init_lr=1e-4,
        weight_decay=1e-5,
        max_epochs=10,
        alpha=0.25,
        gamma=1.5,
    ):
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.init_lr = init_lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.num_classes = 19
        self.alpha = alpha
        self.gamma = gamma

        self.model = self.get_model()

        self.train_acc = Accuracy()
        self.valid_acc = Accuracy()

        self.train_f1 = F1(
            num_classes=self.num_classes, average=None, compute_on_step=False
        )
        self.valid_f1 = F1(
            num_classes=self.num_classes, average=None, compute_on_step=False
        )

        # AUROC returns list type when average=None
        self.train_auroc = AUROC(
            num_classes=self.num_classes, average=None, compute_on_step=False
        )
        self.valid_auroc = AUROC(
            num_classes=self.num_classes, average=None, compute_on_step=False
        )

    def forward(self, x):
        preds = torch.sigmoid(self.model(x))
        return preds

    def training_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.model(images)

        loss = focal_loss(preds, targets, alpha=self.alpha, gamma=self.gamma)

        preds = torch.sigmoid(preds)

        self.log("train_loss", loss)
        self.log("train_acc_step", self.train_acc(preds, targets))
        self.train_f1(preds, targets)
        self.train_auroc(preds, targets)

        return loss

    def training_epoch_end(self, training_step_outputs):
        self.log("train_acc_epoch", self.train_acc.compute())
        train_f1 = self.train_f1.compute()

        try:
            # AUROC returns list type when average=None
            train_auroc = self.train_auroc.compute()
            train_auroc = torch.FloatTensor(train_auroc)
        except ValueError:
            train_auroc = torch.zeros(self.num_classes)

        for i in range(self.num_classes):
            self.log(f"train_f1_{i}", train_f1[i])
            self.log(f"train_auroc_{i}", train_auroc[i])

        self.log("train_f1_mean", torch.mean(train_f1))
        self.log("train_auroc_mean", torch.mean(train_auroc))

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        preds = self.model(images)

        loss = focal_loss(preds, targets, alpha=self.alpha, gamma=self.gamma)

        preds = torch.sigmoid(preds)

        self.log("valid_loss", loss)
        self.log("valid_acc_step", self.valid_acc(preds, targets))
        self.valid_f1(preds, targets)
        self.valid_auroc(preds, targets)

        return loss

    def validation_epoch_end(self, validation_step_outputs):
        self.log("valid_acc_epoch", self.valid_acc.compute())
        valid_f1 = self.valid_f1.compute()

        try:
            # AUROC returns list type when average=None
            valid_auroc = self.valid_auroc.compute()
            valid_auroc = torch.FloatTensor(valid_auroc)
        except ValueError:
            valid_auroc = torch.zeros(self.num_classes)

        for i in range(self.num_classes):
            self.log(f"valid_f1_{i}", valid_f1[i])
            self.log(f"valid_auroc_{i}", valid_auroc[i])

        self.log("valid_f1_mean", torch.mean(valid_f1))
        self.log("valid_auroc_mean", torch.mean(valid_auroc))

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.init_lr, weight_decay=self.weight_decay
        )

        scheduler = CosineAnnealingLR(optimizer, T_max=self.max_epochs)
        print(f"CosineAnnealingLR T_max epochs = {self.max_epochs}")
        return [optimizer], [scheduler]

    def get_model(self):
        model = timm.create_model(
            model_name=self.model_name,
            pretrained=self.pretrained,
            num_classes=self.num_classes,
        )

        return model
