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
    HPAFullDataset,
    HPAFullGCSDataset,
    HPARGYDataset,
    HPARBYDataset,
    HPAGBYDataset,
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


def norm_transform(resize_height, resize_width, norm_mean, norm_std):
    return A.Compose(
        [
            A.Resize(height=resize_height, width=resize_width),
            A.Normalize(mean=norm_mean, std=norm_std),
            ToTensorV2(),
        ]
    )


def base_transform(resize_height, resize_width, norm_mean, norm_std):
    return A.Compose(
        [
            A.Resize(height=resize_height, width=resize_width),
            # scale size(bias=1): (-0.2, 0.2) + 1 = (0.8, 1.2)
            A.RandomScale(scale_limit=(-0.2, 0.2), p=1.0),
            ## scale size(bias=1): (-0.9, 1.0) + 1 = (0.1, 2.0)
            # A.RandomScale(scale_limit=(-0.9, 1.0), p=1.0),
            A.PadIfNeeded(
                min_height=resize_height,
                min_width=resize_width,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=1.0,
            ),
            A.RandomCrop(height=resize_height, width=resize_width, p=1.0),
            A.RandomBrightnessContrast(p=0.8),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
            A.Normalize(mean=norm_mean, std=norm_std),
            ToTensorV2(),
        ]
    )


def no_br_transform(resize_height, resize_width, norm_mean, norm_std):
    return A.Compose(
        [
            A.Resize(height=resize_height, width=resize_width),
            # scale size(bias=1): (-0.2, 0.2) + 1 = (0.8, 1.2)
            A.RandomScale(scale_limit=(-0.2, 0.2), p=1.0),
            ## scale size(bias=1): (-0.9, 1.0) + 1 = (0.1, 2.0)
            # A.RandomScale(scale_limit=(-0.9, 1.0), p=1.0),
            A.PadIfNeeded(
                min_height=resize_height,
                min_width=resize_width,
                border_mode=cv2.BORDER_CONSTANT,
                value=0,
                p=1.0,
            ),
            A.RandomCrop(height=resize_height, width=resize_width, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(border_mode=cv2.BORDER_CONSTANT, value=0, p=0.5),
            A.Normalize(mean=norm_mean, std=norm_std),
            ToTensorV2(),
        ]
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
        train_augmentation="base",
        **kwargs,
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
        self.dataset_cls = HPADataset
        self.setup_message = "Train on HPADataModule."
        self.train_augmentation = train_augmentation

        if self.train_augmentation == "base":
            self.aug_fn = base_transform
        elif self.train_augmentation == "no-br":
            self.aug_fn = no_br_transform
        else:
            self.aug_fn = norm_transform

    def setup(self, stage=None):
        print(self.setup_message)
        self.train_dataset = self.dataset_cls(
            self.dataset_dir, transform=self.get_train_transform()
        )
        self.valid_dataset = self.dataset_cls(
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
        return self.aug_fn(
            resize_height=self.resize_height,
            resize_width=self.resize_width,
            norm_mean=self.norm_mean,
            norm_std=self.norm_std,
        )

    def get_valid_transform(self):
        return norm_transform(
            resize_height=self.resize_height,
            resize_width=self.resize_width,
            norm_mean=self.norm_mean,
            norm_std=self.norm_std,
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

    def __init__(self, dataset_rare_dir="dataset-rare", **kwargs):
        super().__init__(**kwargs)
        self.dataset_rare_dir = dataset_rare_dir
        self.dataset_cls = HPADataset
        self.setup_message = "Train on HPAExtraRareDataModule."

    def setup(self, stage=None):
        print(self.setup_message)
        self.train_hpa_dataset = self.dataset_cls(
            self.dataset_dir, transform=self.get_train_transform()
        )
        self.valid_hpa_dataset = self.dataset_cls(
            self.dataset_dir, transform=self.get_valid_transform()
        )

        self.train_rare_dataset = self.dataset_cls(
            self.dataset_rare_dir, transform=self.get_train_transform()
        )
        self.valid_rare_dataset = self.dataset_cls(
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


class HPAFullDataModule(HPADataModule):
    def __init__(
        self,
        dataset_full_dir="dataset-full",
        train_full_csv="kaggle_2021.tsv",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dataset_full_dir = dataset_full_dir
        self.train_full_csv = train_full_csv
        self.setup_message = "Train on HPAFullDataModule."

    def setup(self, stage=None):
        print(self.setup_message)
        self.train_hpa_dataset = HPADataset(
            self.dataset_dir, transform=self.get_train_transform()
        )
        self.valid_hpa_dataset = HPADataset(
            self.dataset_dir, transform=self.get_valid_transform()
        )

        self.train_full_dataset = HPAFullDataset(
            self.dataset_full_dir,
            transform=self.get_train_transform(),
            train_csv=self.train_full_csv,
        )
        self.valid_full_dataset = HPAFullDataset(
            self.dataset_full_dir,
            transform=self.get_valid_transform(),
            train_csv=self.train_full_csv,
        )

        self.train_dataset = ConcatDataset(
            [self.train_hpa_dataset, self.train_full_dataset]
        )
        self.valid_dataset = ConcatDataset(
            [self.valid_hpa_dataset, self.valid_full_dataset]
        )

        self.train_df = pd.concat(
            [self.train_hpa_dataset.train_df, self.train_full_dataset.train_df],
            axis=0,
        )

        self.train_index, self.valid_index = self.make_fold_index(
            n_splits=self.fold_splits, fold_index=self.fold_index
        )

        self.train_dataset = Subset(self.train_dataset, self.train_index)
        self.valid_dataset = Subset(self.valid_dataset, self.valid_index)


class HPAFullGCSDataModule(HPADataModule):
    def __init__(
        self,
        dataset_full_dir="dataset-full",
        train_full_csv="kaggle_2021.tsv",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.dataset_full_dir = dataset_full_dir
        self.train_full_csv = train_full_csv
        self.setup_message = "Train on HPAFullGCSDataModule."

    def setup(self, stage=None):
        print(self.setup_message)
        self.train_hpa_dataset = HPADataset(
            self.dataset_dir, transform=self.get_train_transform()
        )
        self.valid_hpa_dataset = HPADataset(
            self.dataset_dir, transform=self.get_valid_transform()
        )

        self.train_full_dataset = HPAFullGCSDataset(
            self.dataset_full_dir,
            transform=self.get_train_transform(),
            train_csv=self.train_full_csv,
        )
        self.valid_full_dataset = HPAFullGCSDataset(
            self.dataset_full_dir,
            transform=self.get_valid_transform(),
            train_csv=self.train_full_csv,
        )

        self.train_dataset = ConcatDataset(
            [self.train_hpa_dataset, self.train_full_dataset]
        )
        self.valid_dataset = ConcatDataset(
            [self.valid_hpa_dataset, self.valid_full_dataset]
        )

        self.train_df = pd.concat(
            [self.train_hpa_dataset.train_df, self.train_full_dataset.train_df],
            axis=0,
        )

        self.train_index, self.valid_index = self.make_fold_index(
            n_splits=self.fold_splits, fold_index=self.fold_index
        )

        self.train_dataset = Subset(self.train_dataset, self.train_index)
        self.valid_dataset = Subset(self.valid_dataset, self.valid_index)


class HPARGYDataModule(HPADataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.norm_mean = HPA_RGY_MEAN
        self.norm_std = HPA_RGY_STD
        self.dataset_cls = HPARGYDataset
        self.setup_message = "Train on HPARGYDataModule."


class HPARBYDataModule(HPADataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.norm_mean = HPA_RBY_MEAN
        self.norm_std = HPA_RBY_STD
        self.dataset_cls = HPARBYDataset
        self.setup_message = "Train on HPARBYDataModule."


class HPAGBYDataModule(HPADataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.norm_mean = HPA_GBY_MEAN
        self.norm_std = HPA_GBY_STD
        self.dataset_cls = HPAGBYDataset
        self.setup_message = "Train on HPAGBYDataModule."


class HPARGYExtraRareDataModule(HPAExtraRareDataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.norm_mean = HPA_RGY_MEAN
        self.norm_std = HPA_RGY_STD
        self.dataset_cls = HPARGYDataset
        self.setup_message = "Train on HPARGYExtraRareDataModule."


class HPARBYExtraRareDataModule(HPAExtraRareDataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.norm_mean = HPA_RBY_MEAN
        self.norm_std = HPA_RBY_STD
        self.dataset_cls = HPARBYDataset
        self.setup_message = "Train on HPARBYExtraRareDataModule."


class HPAGBYExtraRareDataModule(HPAExtraRareDataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.norm_mean = HPA_GBY_MEAN
        self.norm_std = HPA_GBY_STD
        self.dataset_cls = HPAGBYDataset
        self.setup_message = "Train on HPAGBYExtraRareDataModule."


class HPASingleLabelExtraRareDataModule(HPAExtraRareDataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset_cls = HPASingleLabelDataset
        self.setup_message = "Train on HPASingleLabelExtraRareDataModule."


class HPARGYSingleLabelExtraRareDataModule(HPAExtraRareDataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.norm_mean = HPA_RGY_MEAN
        self.norm_std = HPA_RGY_STD
        self.dataset_cls = HPARGYSingleLabelDataset
        self.setup_message = "Train on HPARGYSingleLabelExtraRareDataModule."


class HPARBYSingleLabelExtraRareDataModule(HPAExtraRareDataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.norm_mean = HPA_RBY_MEAN
        self.norm_std = HPA_RBY_STD
        self.dataset_cls = HPARBYSingleLabelDataset
        self.setup_message = "Train on HPARBYSingleLabelExtraRareDataModule."


class HPAGBYSingleLabelExtraRareDataModule(HPAExtraRareDataModule):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.norm_mean = HPA_GBY_MEAN
        self.norm_std = HPA_GBY_STD
        self.dataset_cls = HPAGBYSingleLabelDataset
        self.setup_message = "Train on HPAGBYSingleLabelExtraRareDataModule."