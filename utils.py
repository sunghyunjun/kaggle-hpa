from datamodule import *


def add_default_arg(parser):
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument(
        "--dataset-choice",
        default="base",
        choices=[
            "base",
            "extra-rare",
            "full",
            "full-gcs",
            "no-br-extra-rare",
            "single-label-extra-rare",
            "no-br-single-label-extra-rare",
            "rgy-single-label-extra-rare",
            "rby-single-label-extra-rare",
            "gby-single-label-extra-rare",
            "rgy-base",
            "rby-base",
            "gby-base",
            "rgy-extra-rare",
            "rby-extra-rare",
            "gby-extra-rare",
        ],
        help="choose dataset for training. base: default competition's dataset, extra-rare: base + extra-rare dataset",
    )
    parser.add_argument(
        "--lr-finder", action="store_true", help="run learning rate finder"
    )
    parser.add_argument(
        "--dataset-dir", default="dataset", metavar="DIR", help="path to dataset"
    )
    parser.add_argument(
        "--dataset-rare-dir",
        default="dataset-rare",
        metavar="DIR",
        help="path to extra dataset for rare classes",
    )
    parser.add_argument(
        "--dataset-full-dir",
        default="dataset-full",
        metavar="DIR",
        help="path to extra full dataset",
    )
    parser.add_argument(
        "--train-full-csv",
        default="kaggle_2021.tsv",
        metavar="PATH",
        help="path to train csv of full dataset",
    )

    # Pytorch-Lightning Trainer flags
    parser.add_argument(
        "--default-root-dir",
        default=os.getcwd(),
        metavar="DIR",
        help="path to checkpoint saving",
    )

    # Pytorch-Lightning Trainer flags
    parser.add_argument(
        "--gpus", type=int, default=None, help="number of GPU (default: None for CPU)"
    )

    # Pytorch-Lightning Trainer flags
    parser.add_argument(
        "--precision",
        type=int,
        default=32,
        help="precision (default:32, full precision)",
    )

    # Pytorch-Lightning Trainer flags
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="resume from checkpoint in the path",
    )

    # Pytorch-Lightning Trainer flags
    parser.add_argument(
        "--accumulate-grad-batches",
        type=int,
        default=1,
        help="Accumulates grads every k batches",
    )

    # Pytorch-Lightning Trainer flags
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train (default: 10)",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        metavar="N",
        help="input batch size for training (default: 32)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        metavar="N",
        help="how many training processes to use (default: 2)",
    )
    parser.add_argument(
        "--fold-splits",
        type=int,
        default=5,
        metavar="N",
        help="number of folds (default: 5)",
    )
    parser.add_argument(
        "--fold-index",
        type=int,
        default=0,
        metavar="N",
        help="fold index to use for training (default: 0)",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=256,
        metavar="N",
        help="image size for training (default: 256)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="tf_efficientnet_b0",
        metavar="MODEL",
        help="name of model to train (default: 'tf_efficientnet_b0')",
    )
    parser.add_argument(
        "--init-lr",
        type=float,
        default=1e-4,
        metavar="LR",
        help="learning rate (default: 0.0001)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        metavar="DECAY",
        help="weight decay (default: 0.0001)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="focal loss alpha (default: alpha=0.5, gamma=0.0 for bce loss)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.0,
        help="focal loss gamma (default: alpha=0.5, gamma=0.0 for bce loss)",
    )
    parser.add_argument(
        "--smoothing", type=float, default=0.0, help="label smoothing (default: 0.0)"
    )
    parser.add_argument(
        "--mixed-loss",
        action="store_true",
        help="Focal loss at class 1, 11 and BCE loss at others",
    )
    parser.add_argument(
        "--weighted-loss", action="store_true", help="use weighted loss"
    )
    parser.add_argument(
        "--neptune-logger", action="store_true", help="use neptune logger"
    )
    parser.add_argument(
        "--neptune-project", type=str, default=None, help="neptune project"
    )
    parser.add_argument(
        "--neptune-experiment", type=str, default=None, help="neptune experiment name"
    )
    parser.add_argument(
        "--progress-bar-refresh-rate",
        type=int,
        default=1,
        help="how often to refresh progress bar in steps (default: 1)",
    )

    return parser


def create_dm(
    dm_name="base",
    dataset_dir="dataset",
    dataset_rare_dir="dataset-rare",
    dataset_full_dir="dataset-full",
    train_full_csv="kaggle_2021.tsv",
    batch_size=32,
    num_workers=2,
    fold_splits=5,
    fold_index=0,
    image_size=256,
    train_augmentation="base",
):
    cls_dict = {
        "base": HPADataModule,
        "extra-rare": HPAExtraRareDataModule,
        "full": HPAFullDataModule,
        "full-gcs": HPAFullGCSDataModule,
        "no-br-extra-rare": HPAExtraRareDataModule,
        "single-label-extra-rare": HPASingleLabelExtraRareDataModule,
        "no-br-single-label-extra-rare": HPASingleLabelExtraRareDataModule,
        "rgy-single-label-extra-rare": HPARGYSingleLabelExtraRareDataModule,
        "rby-single-label-extra-rare": HPARBYSingleLabelExtraRareDataModule,
        "gby-single-label-extra-rare": HPAGBYSingleLabelExtraRareDataModule,
        "rgy-base": HPARGYDataModule,
        "rby-base": HPARBYDataModule,
        "gby-base": HPAGBYDataModule,
        "rgy-extra-rare": HPARGYExtraRareDataModule,
        "rby-extra-rare": HPARBYExtraRareDataModule,
        "gby-extra-rare": HPAGBYExtraRareDataModule,
    }

    if "no-br" in dm_name:
        train_augmentation = "no-br"

    dm = cls_dict[dm_name](
        dataset_dir=dataset_dir,
        dataset_rare_dir=dataset_rare_dir,
        dataset_full_dir=dataset_full_dir,
        train_full_csv=train_full_csv,
        batch_size=batch_size,
        num_workers=num_workers,
        fold_splits=fold_splits,
        fold_index=fold_index,
        image_size=image_size,
        train_augmentation=train_augmentation,
    )

    return dm