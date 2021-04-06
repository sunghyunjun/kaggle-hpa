import pytorch_lightning as pl
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR


class HPAClassifier(pl.LightningModule):
    def __init__(
        self,
        model_name="tf_efficientnet_b0",
        pretrained=True,
        init_lr=1e-4,
        weight_decay=1e-5,
        max_epochs=10,
    ):
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self.init_lr = init_lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.num_classes = 19

        self.model = self.get_model()

    # TODO
    def forward(self, x):
        target = self.model(x)
        return target

    def training_step(self, batch, batch_idx):
        image, target = batch
        output = self.model(image)

        # loss =

    def training_epoch_end(self, training_step_outputs):
        pass

    def validation_step(self, batch, batch_idx):
        image, target = batch
        output = self.model(image)

        # loss =

    def validation_epoch_end(self, validation_step_outputs):
        pass

    def configure_optimizers(self):
        optimizer = optim.AdamW(
            self.parameters(), lr=self.init_lr, weight_decay=self.weight_decay
        )

        scheduler = CosineAnnealingLR(optimizer, T_max=self.max_eopchs)
        print(f"CosineAnnealingLR T_max epochs = {self.max_epochs}")
        return [optimizer], [scheduler]

    def get_model(self):
        model = timm.create_model(
            model_name=self.model_name,
            pretrained=self.pretrained,
            num_classes=self.num_classes,
        )

        return model
