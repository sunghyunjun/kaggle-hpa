import os

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

HPA_RGB_MEAN = (0.08069728096869229, 0.052413667871535743, 0.053860763980440214)
HPA_RGB_STD = (0.14808042915840622, 0.10954462255820555, 0.15652691280493594)


class HPADataset(Dataset):
    def __init__(self, dataset_dir="dataset", transform=None):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.num_classes = 19
        self.transform = transform
        self.train_csv_path = os.path.join(self.dataset_dir, "train.csv")
        self.load_train_csv()

    def __getitem__(self, index: int):
        image_id, label = self.train_df.iloc[index, :][["ID", "Label"]]
        image = self.get_image(image_id)

        if self.transform is not None:
            image = self.transform(image=image)["image"]
        else:
            image = A.Compose(
                [A.Normalize(mean=HPA_RGB_MEAN, std=HPA_RGB_STD), ToTensorV2()]
            )(image=image)["image"]
            label = torch.as_tensor(label)

        return image, label

    def __len__(self) -> int:
        return len(self.train_df)

    def load_train_csv(self):
        self.train_df = pd.read_csv(self.train_csv_path)

        # Convert "Label", string to numpy.ndarray, remove duplicate label
        self.train_df["Label"] = self.train_df.Label.map(
            lambda x: np.asarray(list(set(x.split("|")))).astype(np.int)
        )

        # One-hot encoded Label, for spliting folds
        self.train_df["Label"] = self.train_df.Label.map(
            lambda x: np.sum(np.eye(self.num_classes)[x].astype(np.int), axis=0)
        )

    def get_image(self, image_id):
        image_red_path = os.path.join(
            self.dataset_dir, "train", image_id + "_red" + ".png"
        )
        image_green_path = os.path.join(
            self.dataset_dir, "train", image_id + "_green" + ".png"
        )
        image_blue_path = os.path.join(
            self.dataset_dir, "train", image_id + "_blue" + ".png"
        )

        image_red = cv2.imread(image_red_path, cv2.IMREAD_GRAYSCALE)
        image_green = cv2.imread(image_green_path, cv2.IMREAD_GRAYSCALE)
        image_blue = cv2.imread(image_blue_path, cv2.IMREAD_GRAYSCALE)

        image = np.dstack([image_red, image_green, image_blue])

        return image


class HPASingleLabelDataset(HPADataset):
    def load_train_csv(self):
        self.train_df = pd.read_csv(self.train_csv_path)

        # remove duplicate label and count labels per image
        self.train_df["single"] = self.train_df.Label.map(
            lambda x: len(set(x.split("|")))
        )

        self.train_df = self.train_df[self.train_df.single == 1]
        self.train_df.reset_index(drop=True, inplace=True)

        # Convert "Label", string to numpy.ndarray, remove duplicate label
        self.train_df["Label"] = self.train_df.Label.map(
            lambda x: np.asarray(list(set(x.split("|")))).astype(np.int)
        )

        # One-hot encoded Label, for spliting folds
        self.train_df["Label"] = self.train_df.Label.map(
            lambda x: np.sum(np.eye(self.num_classes)[x].astype(np.int), axis=0)
        )


class HPARGYDataset(HPADataset):
    def get_image(self, image_id):
        image_red_path = os.path.join(
            self.dataset_dir, "train", image_id + "_red" + ".png"
        )
        image_green_path = os.path.join(
            self.dataset_dir, "train", image_id + "_green" + ".png"
        )
        image_yellow_path = os.path.join(
            self.dataset_dir, "train", image_id + "_yellow" + ".png"
        )

        image_red = cv2.imread(image_red_path, cv2.IMREAD_GRAYSCALE)
        image_green = cv2.imread(image_green_path, cv2.IMREAD_GRAYSCALE)
        image_yellow = cv2.imread(image_yellow_path, cv2.IMREAD_GRAYSCALE)

        image = np.dstack([image_red, image_green, image_yellow])

        return image


class HPARGYSingleLabelDataset(HPARGYDataset):
    def load_train_csv(self):
        self.train_df = pd.read_csv(self.train_csv_path)

        # remove duplicate label and count labels per image
        self.train_df["single"] = self.train_df.Label.map(
            lambda x: len(set(x.split("|")))
        )

        self.train_df = self.train_df[self.train_df.single == 1]
        self.train_df.reset_index(drop=True, inplace=True)

        # Convert "Label", string to numpy.ndarray, remove duplicate label
        self.train_df["Label"] = self.train_df.Label.map(
            lambda x: np.asarray(list(set(x.split("|")))).astype(np.int)
        )

        # One-hot encoded Label, for spliting folds
        self.train_df["Label"] = self.train_df.Label.map(
            lambda x: np.sum(np.eye(self.num_classes)[x].astype(np.int), axis=0)
        )


class HPARBYDataset(HPADataset):
    def get_image(self, image_id):
        image_red_path = os.path.join(
            self.dataset_dir, "train", image_id + "_red" + ".png"
        )
        image_blue_path = os.path.join(
            self.dataset_dir, "train", image_id + "_blue" + ".png"
        )
        image_yellow_path = os.path.join(
            self.dataset_dir, "train", image_id + "_yellow" + ".png"
        )

        image_red = cv2.imread(image_red_path, cv2.IMREAD_GRAYSCALE)
        image_blue = cv2.imread(image_blue_path, cv2.IMREAD_GRAYSCALE)
        image_yellow = cv2.imread(image_yellow_path, cv2.IMREAD_GRAYSCALE)

        image = np.dstack([image_red, image_blue, image_yellow])

        return image


class HPARBYSingleLabelDataset(HPARGYDataset):
    def load_train_csv(self):
        self.train_df = pd.read_csv(self.train_csv_path)

        # remove duplicate label and count labels per image
        self.train_df["single"] = self.train_df.Label.map(
            lambda x: len(set(x.split("|")))
        )

        self.train_df = self.train_df[self.train_df.single == 1]
        self.train_df.reset_index(drop=True, inplace=True)

        # Convert "Label", string to numpy.ndarray, remove duplicate label
        self.train_df["Label"] = self.train_df.Label.map(
            lambda x: np.asarray(list(set(x.split("|")))).astype(np.int)
        )

        # One-hot encoded Label, for spliting folds
        self.train_df["Label"] = self.train_df.Label.map(
            lambda x: np.sum(np.eye(self.num_classes)[x].astype(np.int), axis=0)
        )