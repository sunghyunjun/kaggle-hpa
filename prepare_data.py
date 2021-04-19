from argparse import ArgumentParser
import os
import shutil

import cv2
import numpy as np

from tqdm import tqdm


def main():
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument(
        "--dataset-dir",
        default="dataset",
        metavar="DIR",
        help="path to original raw dataset",
    )
    parser.add_argument(
        "--dataset-outdir",
        default="dataset-resized",
        metavar="DIR",
        help="path to resized dataset",
    )
    parser.add_argument(
        "--image-size", default=1024, type=int, metavar="N", help="resized image size"
    )
    args = parser.parse_args()

    raw_data_dir = os.path.join(args.dataset_dir)
    out_data_dir = os.path.join(args.dataset_outdir)

    os.makedirs(out_data_dir, exist_ok=True)
    os.makedirs(os.path.join(out_data_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(out_data_dir, "test"), exist_ok=True)

    train_images = os.listdir(os.path.join(raw_data_dir, "train"))
    test_images = os.listdir(os.path.join(raw_data_dir, "test"))

    IMAGE_SIZE = args.image_size

    print(f"Resize train images - {IMAGE_SIZE} px png")

    if args.debug:
        pbar = tqdm(train_images[:10])
    else:
        pbar = tqdm(train_images)

    for raw_image in pbar:
        img = cv2.imread(
            os.path.join(raw_data_dir, "train", raw_image), cv2.IMREAD_GRAYSCALE
        )
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(out_data_dir, "train", raw_image), img)

    print(f"Resize test images - {IMAGE_SIZE} px png")

    if args.debug:
        pbar = tqdm(test_images[:10])
    else:
        pbar = tqdm(test_images)

    for raw_image in pbar:
        img = cv2.imread(
            os.path.join(raw_data_dir, "test", raw_image), cv2.IMREAD_GRAYSCALE
        )
        img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        cv2.imwrite(os.path.join(out_data_dir, "test", raw_image), img)

    print(f"Copy csv files")
    shutil.copy(
        os.path.join(raw_data_dir, "sample_submission.csv"), os.path.join(out_data_dir)
    )
    shutil.copy(os.path.join(raw_data_dir, "train.csv"), os.path.join(out_data_dir))

    print(f"Complete resizing")

    # CLI example
    # python prepare_data.py --debug --dataset-dir=dataset-raw --dataset-outdir=dataset-1024 --image-size=1024


if __name__ == "__main__":
    main()