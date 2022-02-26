"""https://github.com/pudae/kaggle-hpa/blob/master/tools/download.py"""

from argparse import ArgumentParser
import io
import os
from multiprocessing import Pool, RLock
import requests
import pathlib
import gzip
import imageio

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


def filter_df(df):
    # Remove all images overlapping with Training set
    df = df[df.in_trainset == False]

    # Remove all images with only labels that are not in this competition
    df = df[~df.Label_idx.isna()]

    colors = ["blue", "red", "green", "yellow"]
    celllines = [
        "A-431",
        "A549",
        "EFO-21",
        "HAP1",
        "HEK 293",
        "HUVEC TERT2",
        "HaCaT",
        "HeLa",
        "PC-3",
        "RH-30",
        "RPTEC TERT1",
        "SH-SY5Y",
        "SK-MEL-30",
        "SiHa",
        "U-2 OS",
        "U-251 MG",
        "hTCEpi",
    ]
    df_17 = df[df.Cellline.isin(celllines)]
    return df_17


def download_and_convert_tifgzip_to_png(url, target_path, image_size=1024):
    """Function to convert .tif.gz to .png and put it in the same folder
    Eg. in Kaggle notebook
    """
    r = requests.get(url)
    f = io.BytesIO(r.content)
    tf = gzip.open(f).read()
    img = imageio.imread(tf, "tiff")
    # imageio.imwrite(target_path, img)

    img = cv2.resize(img, (image_size, image_size), interpolation=cv2.INTER_AREA)
    cv2.imwrite(target_path, img)


def download_single_image(img, save_dir, image_size):
    colors = ["blue", "red", "green", "yellow"]
    try:
        for color in colors:
            img_url = f"{img}_{color}.tif.gz"
            save_path = os.path.join(save_dir, f"{os.path.basename(img)}_{color}.png")
            download_and_convert_tifgzip_to_png(img_url, save_path, image_size)
    except Exception as e:
        print(f"failed to download: {img}")
        print(e)
        return False

    return True


def download(pid, img_list, save_dir, image_size=1024):
    while len(img_list) > 0:
        print(f"try to download {len(img_list)} images")
        try:
            failed_ids = []
            # for img in tqdm(img_list, postfix=f"|p={pid}"):
            #     if not download_single_image(img, save_dir, image_size):
            #         failed_ids.append(img)
            with tqdm(img_list, position=pid + 1, postfix=f"|p={pid}") as pbar:
                for img in pbar:
                    if not download_single_image(img, save_dir, image_size):
                        failed_ids.append(img)
                    pbar.update(1)

            img_list = failed_ids
        except Exception as e:
            print(e)


def main():
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="debug mode")
    args = parser.parse_args()

    num_processes = 2
    image_size = 1024
    csv_path = os.path.join(os.getcwd(), "kaggle_2021.tsv")
    save_dir = os.path.join(os.getcwd(), "dataset-extra-1024/train")

    os.makedirs(save_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    df = filter_df(df)

    ## df.Image: "https://images.proteinatlas.org/10005/921_B9_1"
    url_list = df.Image.tolist()

    if args.debug:
        url_list = url_list[:24]

    url_splits = np.array_split(url_list, num_processes)
    assert sum([len(v) for v in url_splits]) == len(url_list)

    p = Pool(
        processes=num_processes, initargs=(tqdm.get_lock(),), initializer=tqdm.set_lock
    )
    for i, split in enumerate(url_splits):
        p.apply_async(download, args=(i, list(split), save_dir, image_size))
    print("Waiting for all subprocesses done...")
    p.close()
    p.join()
    print("All subprocesses done.")


if __name__ == "__main__":
    main()