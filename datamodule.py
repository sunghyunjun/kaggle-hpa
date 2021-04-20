import os

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import cv2
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Subset, ConcatDataset

from dataset import (
    HPADataset,
    HPARBYSingleLabelDataset,
    HPARGYSingleLabelDataset,
    HPAGBYSingleLabelDataset,
    HPASingleLabelDataset,
    HPA_RGB_MEAN,
    HPA_RGB_STD,
    HPA_RGY_MEAN,
    HPA_RGY_STD,
    HPA_RBY_MEAN,
    HPA_RBY_STD,
    HPA_GBY_MEAN,
    HPA_GBY_STD,
)


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
        self.norm_mean = HPA_RGB_MEAN
        self.norm_std = HPA_RGB_STD

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
                # scale size(bias=1): (-0.2, 0.2) + 1 = (0.8, 1.2)
                A.RandomScale(scale_limit=(-0.2, 0.2), p=1.0),
                ## scale size(bias=1): (-0.9, 1.0) + 1 = (0.1, 2.0)
                # A.RandomScale(scale_limit=(-0.9, 1.0), p=1.0),
                A.PadIfNeeded(
                    min_height=self.resize_height,
                    min_width=self.resize_width,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=1.0,
                ),
                A.RandomCrop(height=self.resize_height, width=self.resize_width, p=1.0),
                A.RandomBrightnessContrast(p=0.8),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.Rotate(border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
                A.Normalize(mean=self.norm_mean, std=self.norm_std),
                ToTensorV2(),
            ]
        )

    def get_valid_transform(self):
        return A.Compose(
            [
                A.Resize(height=self.resize_height, width=self.resize_width),
                A.Normalize(mean=self.norm_mean, std=self.norm_std),
                ToTensorV2(),
            ]
        )

    def make_fold_index(self, n_splits=5, fold_index=0):
        print(f"Fold splits: {n_splits}")
        print(f"Fold index: {fold_index}")
        mskf = MultilabelStratifiedKFold(
            n_splits=n_splits, shuffle=False, random_state=None
        )

        train_fold = []
        valid_fold = []

        X = self.train_df.ID
        y = self.train_df.Label.values
        y = np.stack(y, axis=0)

        for train_index, valid_index in mskf.split(X, y):
            train_fold.append(train_index)
            valid_fold.append(valid_index)

        return train_fold[fold_index], valid_fold[fold_index]


class HPAExtraRareDataModule(HPADataModule):
    """Use Concat Dataset of HPA Competition's default Dataset
    and HPA Full Public Dataset classes 0 and 16 are removed.

    https://www.kaggle.com/c/hpa-single-cell-image-classification/discussion/223822
    https://www.kaggle.com/alexanderriedel/hpa-public-768-excl-0-16?select=hpa_public_excl_0_16_768
    """

    def __init__(
        self,
        dataset_dir="dataset",
        dataset_rare_dir="dataset-rare",
        fold_splits=5,
        fold_index=0,
        batch_size=32,
        num_workers=2,
        image_size=512,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.dataset_rare_dir = dataset_rare_dir
        self.fold_splits = fold_splits
        self.fold_index = fold_index
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resize_height = image_size
        self.resize_width = image_size

    def setup(self, stage=None):
        print("Train on HPAExtraRareDataModule.")
        self.train_hpa_dataset = HPADataset(
            self.dataset_dir, transform=self.get_train_transform()
        )
        self.valid_hpa_dataset = HPADataset(
            self.dataset_dir, transform=self.get_valid_transform()
        )

        self.train_rare_dataset = HPADataset(
            self.dataset_rare_dir, transform=self.get_train_transform()
        )
        self.valid_rare_dataset = HPADataset(
            self.dataset_rare_dir, transform=self.get_valid_transform()
        )

        self.train_dataset = ConcatDataset(
            [self.train_hpa_dataset, self.train_rare_dataset]
        )
        self.valid_dataset = ConcatDataset(
            [self.valid_hpa_dataset, self.valid_rare_dataset]
        )

        self.train_df = pd.concat(
            [self.train_hpa_dataset.train_df, self.train_rare_dataset.train_df],
            axis=0,
        )

        self.train_index, self.valid_index = self.make_fold_index(
            n_splits=self.fold_splits, fold_index=self.fold_index
        )

        self.train_dataset = Subset(self.train_dataset, self.train_index)
        self.valid_dataset = Subset(self.valid_dataset, self.valid_index)


class HPASingleLabelExtraRareDataModule(HPADataModule):
    """Use Concat Dataset of HPA Competition's default Dataset
    and HPA Full Public Dataset classes 0 and 16 are removed.

    https://www.kaggle.com/c/hpa-single-cell-image-classification/discussion/223822
    https://www.kaggle.com/alexanderriedel/hpa-public-768-excl-0-16?select=hpa_public_excl_0_16_768
    """

    def __init__(
        self,
        dataset_dir="dataset",
        dataset_rare_dir="dataset-rare",
        fold_splits=5,
        fold_index=0,
        batch_size=32,
        num_workers=2,
        image_size=512,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.dataset_rare_dir = dataset_rare_dir
        self.fold_splits = fold_splits
        self.fold_index = fold_index
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resize_height = image_size
        self.resize_width = image_size

    def setup(self, stage=None):
        print("Train on HPASingleLabelExtraRareDataModule.")
        self.train_hpa_dataset = HPASingleLabelDataset(
            self.dataset_dir, transform=self.get_train_transform()
        )
        self.valid_hpa_dataset = HPASingleLabelDataset(
            self.dataset_dir, transform=self.get_valid_transform()
        )

        self.train_rare_dataset = HPASingleLabelDataset(
            self.dataset_rare_dir, transform=self.get_train_transform()
        )
        self.valid_rare_dataset = HPASingleLabelDataset(
            self.dataset_rare_dir, transform=self.get_valid_transform()
        )

        self.train_dataset = ConcatDataset(
            [self.train_hpa_dataset, self.train_rare_dataset]
        )
        self.valid_dataset = ConcatDataset(
            [self.valid_hpa_dataset, self.valid_rare_dataset]
        )

        self.train_df = pd.concat(
            [self.train_hpa_dataset.train_df, self.train_rare_dataset.train_df],
            axis=0,
        )

        self.train_index, self.valid_index = self.make_fold_index(
            n_splits=self.fold_splits, fold_index=self.fold_index
        )

        self.train_dataset = Subset(self.train_dataset, self.train_index)
        self.valid_dataset = Subset(self.valid_dataset, self.valid_index)


class HPARGYSingleLabelExtraRareDataModule(HPAExtraRareDataModule):
    def __init__(
        self,
        dataset_dir="dataset",
        dataset_rare_dir="dataset-rare",
        fold_splits=5,
        fold_index=0,
        batch_size=32,
        num_workers=2,
        image_size=512,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.dataset_rare_dir = dataset_rare_dir
        self.fold_splits = fold_splits
        self.fold_index = fold_index
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resize_height = image_size
        self.resize_width = image_size
        self.norm_mean = HPA_RGY_MEAN
        self.norm_std = HPA_RGY_STD

    def setup(self, stage=None):
        print("Train on HPARGYSingleLabelExtraRareDataModule.")
        self.train_hpa_dataset = HPARGYSingleLabelDataset(
            self.dataset_dir, transform=self.get_train_transform()
        )
        self.valid_hpa_dataset = HPARGYSingleLabelDataset(
            self.dataset_dir, transform=self.get_valid_transform()
        )

        self.train_rare_dataset = HPARGYSingleLabelDataset(
            self.dataset_rare_dir, transform=self.get_train_transform()
        )
        self.valid_rare_dataset = HPARGYSingleLabelDataset(
            self.dataset_rare_dir, transform=self.get_valid_transform()
        )

        self.train_dataset = ConcatDataset(
            [self.train_hpa_dataset, self.train_rare_dataset]
        )
        self.valid_dataset = ConcatDataset(
            [self.valid_hpa_dataset, self.valid_rare_dataset]
        )

        self.train_df = pd.concat(
            [self.train_hpa_dataset.train_df, self.train_rare_dataset.train_df],
            axis=0,
        )

        self.train_index, self.valid_index = self.make_fold_index(
            n_splits=self.fold_splits, fold_index=self.fold_index
        )

        self.train_dataset = Subset(self.train_dataset, self.train_index)
        self.valid_dataset = Subset(self.valid_dataset, self.valid_index)


class HPARBYSingleLabelExtraRareDataModule(HPAExtraRareDataModule):
    def __init__(
        self,
        dataset_dir="dataset",
        dataset_rare_dir="dataset-rare",
        fold_splits=5,
        fold_index=0,
        batch_size=32,
        num_workers=2,
        image_size=512,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.dataset_rare_dir = dataset_rare_dir
        self.fold_splits = fold_splits
        self.fold_index = fold_index
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resize_height = image_size
        self.resize_width = image_size
        self.norm_mean = HPA_RBY_MEAN
        self.norm_std = HPA_RBY_STD

    def setup(self, stage=None):
        print("Train on HPARBYSingleLabelExtraRareDataModule.")
        self.train_hpa_dataset = HPARBYSingleLabelDataset(
            self.dataset_dir, transform=self.get_train_transform()
        )
        self.valid_hpa_dataset = HPARBYSingleLabelDataset(
            self.dataset_dir, transform=self.get_valid_transform()
        )

        self.train_rare_dataset = HPARBYSingleLabelDataset(
            self.dataset_rare_dir, transform=self.get_train_transform()
        )
        self.valid_rare_dataset = HPARBYSingleLabelDataset(
            self.dataset_rare_dir, transform=self.get_valid_transform()
        )

        self.train_dataset = ConcatDataset(
            [self.train_hpa_dataset, self.train_rare_dataset]
        )
        self.valid_dataset = ConcatDataset(
            [self.valid_hpa_dataset, self.valid_rare_dataset]
        )

        self.train_df = pd.concat(
            [self.train_hpa_dataset.train_df, self.train_rare_dataset.train_df],
            axis=0,
        )

        self.train_index, self.valid_index = self.make_fold_index(
            n_splits=self.fold_splits, fold_index=self.fold_index
        )

        self.train_dataset = Subset(self.train_dataset, self.train_index)
        self.valid_dataset = Subset(self.valid_dataset, self.valid_index)


class HPAGBYSingleLabelExtraRareDataModule(HPAExtraRareDataModule):
    def __init__(
        self,
        dataset_dir="dataset",
        dataset_rare_dir="dataset-rare",
        fold_splits=5,
        fold_index=0,
        batch_size=32,
        num_workers=2,
        image_size=512,
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.dataset_rare_dir = dataset_rare_dir
        self.fold_splits = fold_splits
        self.fold_index = fold_index
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resize_height = image_size
        self.resize_width = image_size
        self.norm_mean = HPA_GBY_MEAN
        self.norm_std = HPA_GBY_STD

    def setup(self, stage=None):
        print("Train on HPAGBYSingleLabelExtraRareDataModule.")
        self.train_hpa_dataset = HPAGBYSingleLabelDataset(
            self.dataset_dir, transform=self.get_train_transform()
        )
        self.valid_hpa_dataset = HPAGBYSingleLabelDataset(
            self.dataset_dir, transform=self.get_valid_transform()
        )

        self.train_rare_dataset = HPAGBYSingleLabelDataset(
            self.dataset_rare_dir, transform=self.get_train_transform()
        )
        self.valid_rare_dataset = HPAGBYSingleLabelDataset(
            self.dataset_rare_dir, transform=self.get_valid_transform()
        )

        self.train_dataset = ConcatDataset(
            [self.train_hpa_dataset, self.train_rare_dataset]
        )
        self.valid_dataset = ConcatDataset(
            [self.valid_hpa_dataset, self.valid_rare_dataset]
        )

        self.train_df = pd.concat(
            [self.train_hpa_dataset.train_df, self.train_rare_dataset.train_df],
            axis=0,
        )

        self.train_index, self.valid_index = self.make_fold_index(
            n_splits=self.fold_splits, fold_index=self.fold_index
        )

        self.train_dataset = Subset(self.train_dataset, self.train_index)
        self.valid_dataset = Subset(self.valid_dataset, self.valid_index)