from argparse import ArgumentParser
import os

import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint

from datamodule import (
    HPADataModule,
    HPAExtraRareDataModule,
    HPARGYDataModule,
    HPARBYDataModule,
    HPAGBYDataModule,
    HPARGYExtraRareDataModule,
    HPARBYExtraRareDataModule,
    HPAGBYExtraRareDataModule,
    HPASingleLabelExtraRareDataModule,
    HPARGYSingleLabelExtraRareDataModule,
    HPARBYSingleLabelExtraRareDataModule,
    HPAGBYSingleLabelExtraRareDataModule,
)
from models import HPAClassifier


def main():
    # ----------
    # seed
    # ----------
    pl.seed_everything(0)

    # ----------
    # args
    # ----------
    parser = ArgumentParser(description="PyTorch HPA Training")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument(
        "--dataset-choice",
        default="base",
        choices=[
            "base",
            "extra-rare",
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

    args = parser.parse_args()

    # ----------
    # for debug
    # ----------
    if args.debug:
        args.max_epochs = 1
        args.batch_size = 2
        args.limit_train_batches = 3
        args.limit_val_batches = 3

    # ----------
    # data
    # ----------
    if args.dataset_choice == "extra-rare":
        dm = HPAExtraRareDataModule(
            dataset_dir=args.dataset_dir,
            dataset_rare_dir=args.dataset_rare_dir,
            batch_size=args.batch_size,
            num_workers=args.workers,
            fold_splits=args.fold_splits,
            fold_index=args.fold_index,
            image_size=args.image_size,
        )
    elif args.dataset_choice == "no-br-extra-rare":
        dm = HPAExtraRareDataModule(
            dataset_dir=args.dataset_dir,
            dataset_rare_dir=args.dataset_rare_dir,
            batch_size=args.batch_size,
            num_workers=args.workers,
            fold_splits=args.fold_splits,
            fold_index=args.fold_index,
            image_size=args.image_size,
            train_augmentation="no-br",
        )
    elif args.dataset_choice == "single-label-extra-rare":
        dm = HPASingleLabelExtraRareDataModule(
            dataset_dir=args.dataset_dir,
            dataset_rare_dir=args.dataset_rare_dir,
            batch_size=args.batch_size,
            num_workers=args.workers,
            fold_splits=args.fold_splits,
            fold_index=args.fold_index,
            image_size=args.image_size,
        )
    elif args.dataset_choice == "no-br-single-label-extra-rare":
        dm = HPASingleLabelExtraRareDataModule(
            dataset_dir=args.dataset_dir,
            dataset_rare_dir=args.dataset_rare_dir,
            batch_size=args.batch_size,
            num_workers=args.workers,
            fold_splits=args.fold_splits,
            fold_index=args.fold_index,
            image_size=args.image_size,
            train_augmentation="no-br",
        )
    elif args.dataset_choice == "rgy-single-label-extra-rare":
        dm = HPARGYSingleLabelExtraRareDataModule(
            dataset_dir=args.dataset_dir,
            dataset_rare_dir=args.dataset_rare_dir,
            batch_size=args.batch_size,
            num_workers=args.workers,
            fold_splits=args.fold_splits,
            fold_index=args.fold_index,
            image_size=args.image_size,
        )
    elif args.dataset_choice == "rby-single-label-extra-rare":
        dm = HPARBYSingleLabelExtraRareDataModule(
            dataset_dir=args.dataset_dir,
            dataset_rare_dir=args.dataset_rare_dir,
            batch_size=args.batch_size,
            num_workers=args.workers,
            fold_splits=args.fold_splits,
            fold_index=args.fold_index,
            image_size=args.image_size,
        )
    elif args.dataset_choice == "gby-single-label-extra-rare":
        dm = HPAGBYSingleLabelExtraRareDataModule(
            dataset_dir=args.dataset_dir,
            dataset_rare_dir=args.dataset_rare_dir,
            batch_size=args.batch_size,
            num_workers=args.workers,
            fold_splits=args.fold_splits,
            fold_index=args.fold_index,
            image_size=args.image_size,
        )
    elif args.dataset_choice == "rgy-base":
        dm = HPARGYDataModule(
            dataset_dir=args.dataset_dir,
            batch_size=args.batch_size,
            num_workers=args.workers,
            fold_splits=args.fold_splits,
            fold_index=args.fold_index,
            image_size=args.image_size,
        )
    elif args.dataset_choice == "rby-base":
        dm = HPARBYDataModule(
            dataset_dir=args.dataset_dir,
            batch_size=args.batch_size,
            num_workers=args.workers,
            fold_splits=args.fold_splits,
            fold_index=args.fold_index,
            image_size=args.image_size,
        )
    elif args.dataset_choice == "gby-base":
        dm = HPAGBYDataModule(
            dataset_dir=args.dataset_dir,
            batch_size=args.batch_size,
            num_workers=args.workers,
            fold_splits=args.fold_splits,
            fold_index=args.fold_index,
            image_size=args.image_size,
        )
    elif args.dataset_choice == "rgy-extra-rare":
        dm = HPARGYExtraRareDataModule(
            dataset_dir=args.dataset_dir,
            dataset_rare_dir=args.dataset_rare_dir,
            batch_size=args.batch_size,
            num_workers=args.workers,
            fold_splits=args.fold_splits,
            fold_index=args.fold_index,
            image_size=args.image_size,
        )
    elif args.dataset_choice == "rby-extra-rare":
        dm = HPARBYExtraRareDataModule(
            dataset_dir=args.dataset_dir,
            dataset_rare_dir=args.dataset_rare_dir,
            batch_size=args.batch_size,
            num_workers=args.workers,
            fold_splits=args.fold_splits,
            fold_index=args.fold_index,
            image_size=args.image_size,
        )
    elif args.dataset_choice == "gby-extra-rare":
        dm = HPAGBYExtraRareDataModule(
            dataset_dir=args.dataset_dir,
            dataset_rare_dir=args.dataset_rare_dir,
            batch_size=args.batch_size,
            num_workers=args.workers,
            fold_splits=args.fold_splits,
            fold_index=args.fold_index,
            image_size=args.image_size,
        )
    else:
        dm = HPADataModule(
            dataset_dir=args.dataset_dir,
            batch_size=args.batch_size,
            num_workers=args.workers,
            fold_splits=args.fold_splits,
            fold_index=args.fold_index,
            image_size=args.image_size,
        )

    # ----------
    # model
    # ----------
    dm.setup()

    model = HPAClassifier(
        model_name=args.model,
        init_lr=args.init_lr,
        weight_decay=args.weight_decay,
        max_epochs=args.max_epochs,
        alpha=args.alpha,
        gamma=args.gamma,
        mixed_loss=args.mixed_loss,
        smoothing=args.smoothing,
        weighted_loss=args.weighted_loss,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid_loss",
        filename="hpa-clf-{epoch:03d}-{valid_loss:.6f}",
        save_last=True,
        save_top_k=3,
        mode="min",
    )

    # ----------
    # logger
    # ----------
    if args.neptune_logger:
        logger = NeptuneLogger(
            api_key=os.environ["NEPTUNE_API_TOKEN"],
            project_name=args.neptune_project,
            experiment_name=args.neptune_experiment,
            params=args.__dict__,
            upload_source_files="*.py",
        )
    else:
        # Use default logger, TenserBoardLogger
        logger = True

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # ----------
    # training
    # ----------
    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=[checkpoint_callback, lr_monitor], logger=logger
    )

    if args.lr_finder:
        lr_finder = trainer.tuner.lr_find(model=model, datamodule=dm)
        fig = lr_finder.plot(suggest=True)
        fig.savefig("./suggested_lr.png")
        suggested_lr = lr_finder.suggestion()
        print(suggested_lr)
    else:
        trainer.fit(model, dm)


if __name__ == "__main__":
    main()