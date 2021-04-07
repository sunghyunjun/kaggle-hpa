# kaggle-hpa

```bash
mkdir dataset
mkdir dataset/train

kaggle datasets download -d phalanx/hpa-512512
unzip hpa-512512.zip -d dataset/train

kaggle competitions download hpa-single-cell-image-classification -f train.csv
mv train.csv dataset
```
