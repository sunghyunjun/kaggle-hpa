# kaggle-hpa

Code for 46th place solution in [Kaggle Human Protein Atlas - Single Cell Classification](https://www.kaggle.com/c/hpa-single-cell-image-classification)

*Read this in other languages: [English](https://github.com/sunghyunjun/kaggle-hpa/blob/main/README.md), [한국어](https://github.com/sunghyunjun/kaggle-hpa/blob/main/README.ko.md)*

## Summary

"Simple Image Level Multilabel Classifier"

The label was determined by applying a classifier to the single cell mask obtained by HPA-Cell-Segmentation.

Classifier was trained using the full dataset.

## Tools

- Colab Pro, GCE, Tesla V100 16GB single GPU
- GCS
- Pytorch Lightning
- Neptune
- Kaggle API

## Dataset

I used both the Competitions default dataset and the extra dataset.

[HPA 512 PNG Dataset](https://www.kaggle.com/phalanx/hpa-512512) by [@phalanx](https://www.kaggle.com/phalanx)

[HPA 768 PNG Dataset](https://www.kaggle.com/phalanx/hpa-768768) by [@phalanx](https://www.kaggle.com/phalanx)

[HPA 1024 PNG Dataset](https://www.kaggle.com/sunghyunjun/hpa-1024-png-dataset)

[HPA Public Data 768x768 "rare classes" dataset](https://www.kaggle.com/c/hpa-single-cell-image-classification/discussion/223822) by [@Alexander Riedel](https://www.kaggle.com/alexanderriedel)

The extra dataset was downloaded by referring to the public note.
Images saved to 768px png. The size is approximately 200 GB.

[HPA public data download and HPACellSeg](https://www.kaggle.com/lnhtrang/hpa-public-data-download-and-hpacellseg)

## Validation

MultilabelStratifiedKFold, 5-fold split was used.

The performance of Multilabel Classifier was verified with Macro-F1, Micro-F1 Score.

[iterative-stratification](https://github.com/trent-b/iterative-stratification)

## Model training

3-channel RGB images

The image size is 1024px, and trained with the following dataset.

- 1024px Competition default dataset + 768px rare classes dataset(1024 resized)
- 1024px Competition default dataset + 768px extra dataset(1024 resized)
- AdamW
- CosineAnnealingLR
- epochs = 5 for full, 10 for rare

bce, focal loss was used.

|model|dataset|folds|loss|batch_size|init_lr|weight_decay|macro F1|micro F1|public LB|private LB|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|efficientnet_b0|full|2 of 5|bce|16|6.0e-4|1.0e-5|0.7663|0.8171|0.454|0.429|
|efficientnet_b0|rare classes|single|bce|16|6.0e-4|1.0e-5|0.8154|0.8368|0.394|0.360|
|seresnext26d_32x4d|full|single|alpha=0.75, gamma=0.0|14|6.5e-5|1.0e-5|0.7317|0.7956|0.381|0.335|
|**final ensemble**|||||||||**0.471**|**0.433**|

## Segmentation

HPA-Cell-Segmentation was used, and the speed was improved by referring to @deoxy's notebook.

The input image was resized by 1/4, and the CellSegmentator scale_factor=1.0.

The related values of the label_cell function have been adjusted to 1/4.

[HPA-Cell-Segmentation](https://github.com/CellProfiling/HPA-Cell-Segmentation)

[Faster HPA Cell Segmentation](https://www.kaggle.com/linshokaku/faster-hpa-cell-segmentation)
by [@deoxy](https://www.kaggle.com/linshokaku)

## Augmentation

```python
A.Compose(
    [
        A.Resize(height=resize_height, width=resize_width),
        A.RandomScale(scale_limit=(-0.2, 0.2), p=1.0),
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
```

## TTA 4x

HorizontalFlip, VerticalFlip, Resize 0.8, Resize 1.2

## What did not work

- Label Smoothing

- pos/neg balanced weighted loss

    X. Wang, Y. Peng, L. Lu, Z. Lu, M. Bagheri, and R. M. Summers.
(Dec. 2017). "ChestX-ray8: Hospital-scale chest X-ray database and
benchmarks on weakly-supervised classification and localization of common thorax diseases.", (p. 5) [https://arxiv.org/abs/1705.02315](https://arxiv.org/abs/1705.02315)
