import os

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset

from dataset import HPADataset, HPA_RGB_MEAN, HPA_RGB_STD


class HPADataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_dir="dataset",
        fold_splits=5,
        fold_index=0,
        batch_size=32,
        num_workers=2,
        image_size=512,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.fold_splits = fold_splits
        self.fold_index = fold_index
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resize_height = image_size
        self.resize_width = image_size

    def setup(self, stage=None):
        self.train_dataset = HPADataset(
            self.dataset_dir, transform=self.get_train_transform()
        )
        self.valid_dataset = HPADataset(
            self.dataset_dir, transform=self.get_valid_transform()
        )

        self.train_df = self.train_dataset.train_df

        self.train_index, self.valid_index = self.make_fold_index(
            n_splits=self.fold_splits, fold_index=self.fold_index
        )

        self.train_dataset = Subset(self.train_dataset, self.train_index)
        self.valid_dataset = Subset(self.valid_dataset, self.valid_index)

    def train_dataloader(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return train_loader

    def val_dataloader(self):
        valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
        return valid_loader

    def get_train_transform(self):
        return A.Compose(
            [
                A.Resize(height=self.resize_height, width=self.resize_width),
                # A.RandomScale(scale_limit=(-0.9, 1.0), p=1.0),
                # A.PadIfNeeded(
                #     min_height=self.resize_height,
                #     min_width=self.resize_width,
                #     border_mode=cv2.BORDER_CONSTANT,
                #     value=0,
                #     p=1.0,
                # ),
                # A.RandomCrop(height=self.resize_height, width=self.resize_width, p=1.0),
                # A.RandomBrightnessContrast(p=0.8),
                # # A.ChannelDropout(p=0.5),
                # A.OneOf(
                #     [
                #         A.MotionBlur(p=0.5),
                #         A.MedianBlur(p=0.5),
                #         A.GaussianBlur(p=0.5),
                #         A.GaussNoise(p=0.5),
                #     ],
                #     p=0.5,
                # ),
                # A.HorizontalFlip(p=0.5),
                A.Normalize(mean=HPA_RGB_MEAN, std=HPA_RGB_STD),
                ToTensorV2(),
            ]
        )

    def get_valid_transform(self):
        return A.Compose(
            [
                A.Resize(height=self.resize_height, width=self.resize_width),
                A.Normalize(mean=HPA_RGB_MEAN, std=HPA_RGB_STD),
                ToTensorV2(),
            ]
        )

    def make_fold_index(self, n_splits=5, fold_index=0):
        print(f"Fold splits: {n_splits}")
        print(f"Fold index: {fold_index}")
        mskf = MultilabelStratifiedKFold(n_splits=n_splits, shuffle=False)

        train_fold = []
        valid_fold = []

        X = self.train_df.ID
        y = self.train_df.Label_onehot.values
        y = np.stack(y, axis=0)

        for train_index, valid_index in mskf.split(X, y):
            train_fold.append(train_index)
            valid_fold.append(valid_index)

        return train_fold[fold_index], valid_fold[fold_index]
