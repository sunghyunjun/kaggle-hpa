# kaggle-hpa

Code for 46th place solution in [Kaggle Human Protein Atlas - Single Cell Classification](https://www.kaggle.com/c/hpa-single-cell-image-classification)

*Read this in other languages: [English](https://github.com/sunghyunjun/kaggle-hpa/blob/main/README.md), [한국어](https://github.com/sunghyunjun/kaggle-hpa/blob/main/README.ko.md)*

## Summary

"Simple Image Level Multilabel Classifier"

HPA-Cell-Segmentation으로 구해진 Single Cell Mask에 Classifier를 적용시켜 Label을 판별하였습니다.

Classifier는 Full dataset을 이용해 훈련하였습니다.

## Tools

- Colab Pro, GCE, Tesla V100 16GB single GPU
- GCS
- Pytorch Lightning
- Neptune
- Kaggle API

## Dataset

Competitions default dataset과 extra dataset을 모두 사용하였습니다.

[HPA 512 PNG Dataset](https://www.kaggle.com/phalanx/hpa-512512) by [@phalanx](https://www.kaggle.com/phalanx)

[HPA 768 PNG Dataset](https://www.kaggle.com/phalanx/hpa-768768) by [@phalanx](https://www.kaggle.com/phalanx)

[HPA 1024 PNG Dataset](https://www.kaggle.com/sunghyunjun/hpa-1024-png-dataset)

[HPA Public Data 768x768 "rare classes" dataset](https://www.kaggle.com/c/hpa-single-cell-image-classification/discussion/223822) by [@Alexander Riedel](https://www.kaggle.com/alexanderriedel)

extra dataset은 public note를 참고하여 다운로드 하였으며,
768px png로 변환하여 구축하였습니다. 용량은 대략 200GB 정도입니다.

[HPA public data download and HPACellSeg](https://www.kaggle.com/lnhtrang/hpa-public-data-download-and-hpacellseg)

## Validation

MultilabelStratifiedKFold를 사용하였으며, 5 fold로 데이터 세트를 구성하였습니다.

Multilabel Classifier의 성능은 Macro-F1, Micro-F1 Score로 검증하였습니다.

[iterative-stratification](https://github.com/trent-b/iterative-stratification)

## Model training

image size는 1024px 이며 다음의 데이터셋으로 훈련하였습니다.

- 1024px Competition default dataset + 768px rare classes dataset(1024 resized)
- 1024px Competition default dataset + 768px extra dataset(1024 resized)

loss는 bce, focal loss를 사용하였습니다.

|model|dataset|folds|loss|batch_size|init_lr|weight_decay|macro F1|micro F1|public LB|private LB|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|efficientnet_b0|full|2 of 5|bce|16|6.0e-4|1.0e-5|0.7663|0.8171|0.454|0.429|
|efficientnet_b0|rare classes|single|bce|16|6.0e-4|1.0e-5|0.8154|0.8368|0.394|0.360|
|seresnext26d_32x4d|full|single|alpha=0.75, gamma=0.0|14|6.5e-5|1.0e-5|0.7317|0.7956|checking|checking|
|**final ensemble**|||||||||**0.471**|**0.433**|

## Segmentation

HPA-Cell-Segmentation을 사용하였으며, @deoxy의 notebook을 참고하여 속도를 개선했습니다.
input image를 1/4 resize 하였고, CellSegmentator의 scale_factor=1.0으로 하였습니다.
label_cell function의 관련 값들을 1/4로 조절하였습니다.

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
