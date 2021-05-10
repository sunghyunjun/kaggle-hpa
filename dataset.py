import io
import math
import os
import typing
from urllib.parse import urlparse
from urllib3.exceptions import ProtocolError

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from PIL import Image
from PIL.Image import Image as ImageType
from google.cloud import storage
from google.api_core import retry
from google.api_core.retry import Retry

HPA_RED_MEAN = 0.08069728096869229
HPA_GREEN_MEAN = 0.052413667871535743
HPA_BLUE_MEAN = 0.053860763980440214
HPA_YELLOW_MEAN = 0.08102950870133106

HPA_RED_STD = 0.14808042915840622
HPA_GREEN_STD = 0.10954462255820555
HPA_BLUE_STD = 0.15652691280493594
HPA_YELLOW_STD = 0.27288042696564363

HPA_RGB_MEAN = (HPA_RED_MEAN, HPA_GREEN_MEAN, HPA_BLUE_MEAN)
HPA_RGB_STD = (HPA_RED_STD, HPA_GREEN_STD, HPA_BLUE_STD)

HPA_RGY_MEAN = (HPA_RED_MEAN, HPA_GREEN_MEAN, HPA_YELLOW_MEAN)
HPA_RGY_STD = (HPA_RED_STD, HPA_GREEN_STD, HPA_YELLOW_STD)

HPA_RBY_MEAN = (HPA_RED_MEAN, HPA_BLUE_MEAN, HPA_YELLOW_MEAN)
HPA_RBY_STD = (HPA_RED_STD, HPA_BLUE_STD, HPA_YELLOW_STD)

HPA_GBY_MEAN = (HPA_GREEN_MEAN, HPA_BLUE_MEAN, HPA_YELLOW_MEAN)
HPA_GBY_STD = (HPA_GREEN_STD, HPA_BLUE_STD, HPA_YELLOW_STD)

predicate = retry.if_exception_type(ConnectionResetError, ProtocolError)


@Retry()
def gcs_pil_loader(uri) -> ImageType:
    uri = urlparse(uri)
    client = storage.Client()
    bucket = client.get_bucket(uri.netloc)
    b = bucket.blob(uri.path[1:], chunk_size=None)
    image_pil = Image.open(io.BytesIO(b.download_as_string()))
    image_pil = image_pil.convert("I")
    image = np.asarray(image_pil, dtype=np.uint16)
    image = image / 256
    image = image.astype(np.uint8)
    return image


@Retry(predicate=predicate)
def gcs_cv_loader(uri):
    uri = urlparse(uri)
    client = storage.Client()
    bucket = client.get_bucket(uri.netloc)
    b = bucket.blob(uri.path[1:], chunk_size=None)
    nparr = np.frombuffer(b.download_as_string(), np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    return image


class StringArray:
    # def __init__(
    #     self,
    #     strings: typing.List[str],
    #     encoding: typing.Literal["ascii", "utf_16_le", "utf_32_le"] = "utf_16_le",
    # ):
    def __init__(
        self,
        strings: typing.List[str],
        encoding="utf_16_le",
    ):
        strings = list(strings)
        self.encoding = encoding
        self.multiplier = dict(ascii=1, utf_16_le=2, utf_32_le=4)[encoding]
        self.data = torch.ByteTensor(
            torch.ByteStorage.from_buffer("".join(strings).encode(encoding))
        )
        self.cumlen = (
            torch.LongTensor(list(map(len, strings)))
            .cumsum(dim=0)
            .mul_(self.multiplier)
        )
        assert int(self.cumlen[-1]) == len(
            self.data
        ), f"[{encoding}] is not enough to hold characters, use a larger character class"

    def __getitem__(self, i):
        return bytes(
            self.data[(self.cumlen[i - 1] if i >= 1 else 0) : self.cumlen[i]]
        ).decode(self.encoding)

    def __len__(self):
        return len(self.cumlen)

    def tolist(self):
        data_bytes, cumlen = bytes(self.data), self.cumlen.tolist()
        return [data_bytes[0 : cumlen[0]].decode(self.encoding)] + [
            data_bytes[start:end].decode(self.encoding)
            for start, end in zip(cumlen[:-1], cumlen_mul[1:])
        ]


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


class HPAFullDataset(HPADataset):
    """HPA FullDataset. Competition's default public train set were excluded."""

    def __init__(
        self,
        dataset_dir="dataset-full",
        transform=None,
        train_csv="kaggle_2021.tsv",
    ):
        super(HPADataset, self).__init__()
        self.dataset_dir = dataset_dir
        self.num_classes = 19
        self.transform = transform
        self.train_csv_path = train_csv
        self.load_train_csv()
        cv2.setNumThreads(0)

    def load_train_csv(self):
        self.train_df = pd.read_csv(self.train_csv_path)

        # Drop nan Label_idx
        self.train_df = self.train_df[~self.train_df.Label_idx.isnull()].reset_index(
            drop=True
        )

        # Drop competition's default public train dataset
        self.train_df = self.train_df[~self.train_df.in_trainset].reset_index(drop=True)

        self.train_df["ID"] = self.train_df.Image.map(lambda x: x.split("/")[-1])
        self.train_df["Label"] = self.train_df.Label_idx

        self.train_df = self.train_df[["ID", "Label"]]

        # Convert "Label", string to numpy.ndarray, remove duplicate label
        self.train_df["Label"] = self.train_df.Label.map(
            lambda x: np.asarray(list(set(x.split("|")))).astype(np.int)
        )

        # One-hot encoded Label, for spliting folds
        self.train_df["Label"] = self.train_df.Label.map(
            lambda x: np.sum(np.eye(self.num_classes)[x].astype(np.int), axis=0)
        )

    def get_image(self, image_id):
        image_red_path = os.path.join(self.dataset_dir, image_id + "_red" + ".png")
        image_green_path = os.path.join(self.dataset_dir, image_id + "_green" + ".png")
        image_blue_path = os.path.join(self.dataset_dir, image_id + "_blue" + ".png")

        image_red = cv2.imread(image_red_path, cv2.IMREAD_GRAYSCALE)
        image_green = cv2.imread(image_green_path, cv2.IMREAD_GRAYSCALE)
        image_blue = cv2.imread(image_blue_path, cv2.IMREAD_GRAYSCALE)

        image = np.dstack([image_red, image_green, image_blue])

        return image


class HPAFullGCSDataset(Dataset):
    def __init__(
        self,
        dataset_dir="dataset-full",
        transform=None,
        train_csv="kaggle_2021.tsv",
    ):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.num_classes = 19
        self.transform = transform
        self.train_csv_path = train_csv
        self.load_train_csv()
        self.convert_index()
        cv2.setNumThreads(0)

    def __getitem__(self, index: int):
        # image_id, label = self.train_df.iloc[index, :][["ID", "Label"]]
        image_id = self.image_ids[index]
        label = self.labels[index]
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

        # Drop nan Label_idx
        self.train_df = self.train_df[~self.train_df.Label_idx.isnull()].reset_index(
            drop=True
        )

        # Drop competition's default public train dataset
        self.train_df = self.train_df[~self.train_df.in_trainset].reset_index(drop=True)

        self.train_df["ID"] = self.train_df.Image.map(lambda x: x.split("/")[-1])
        self.train_df["Label"] = self.train_df.Label_idx

        self.train_df = self.train_df[["ID", "Label"]]

        # Convert "Label", string to numpy.ndarray, remove duplicate label
        self.train_df["Label"] = self.train_df.Label.map(
            lambda x: np.asarray(list(set(x.split("|")))).astype(np.int)
        )

        # One-hot encoded Label, for spliting folds
        self.train_df["Label"] = self.train_df.Label.map(
            lambda x: np.sum(np.eye(self.num_classes)[x].astype(np.int), axis=0)
        )

    def convert_index(self):
        id_list = self.train_df.ID.tolist()
        self.image_ids = StringArray(id_list)

        label_list = self.train_df.Label.tolist()
        self.labels = np.stack(label_list)

    def get_image(self, image_id):
        image_red_path = os.path.join(self.dataset_dir, image_id + "_red" + ".png")
        image_green_path = os.path.join(self.dataset_dir, image_id + "_green" + ".png")
        image_blue_path = os.path.join(self.dataset_dir, image_id + "_blue" + ".png")

        image_red = gcs_cv_loader(image_red_path)
        image_green = gcs_cv_loader(image_green_path)
        image_blue = gcs_cv_loader(image_blue_path)

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


class HPAGBYDataset(HPADataset):
    def get_image(self, image_id):
        image_green_path = os.path.join(
            self.dataset_dir, "train", image_id + "_green" + ".png"
        )
        image_blue_path = os.path.join(
            self.dataset_dir, "train", image_id + "_blue" + ".png"
        )
        image_yellow_path = os.path.join(
            self.dataset_dir, "train", image_id + "_yellow" + ".png"
        )

        image_green = cv2.imread(image_green_path, cv2.IMREAD_GRAYSCALE)
        image_blue = cv2.imread(image_blue_path, cv2.IMREAD_GRAYSCALE)
        image_yellow = cv2.imread(image_yellow_path, cv2.IMREAD_GRAYSCALE)

        image = np.dstack([image_green, image_blue, image_yellow])

        return image


class HPAGBYSingleLabelDataset(HPAGBYDataset):
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