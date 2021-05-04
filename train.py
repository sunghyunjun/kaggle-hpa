from argparse import ArgumentParser
import os

import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint

from models import HPAClassifier
from utils import add_default_arg, create_dm


def main():
    # ----------
    # seed
    # ----------
    pl.seed_everything(0)

    # ----------
    # args
    # ----------
    parser = ArgumentParser(description="PyTorch HPA Training")
    parser = add_default_arg(parser)
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
    dm = create_dm(
        dm_name=args.dataset_choice,
        dataset_dir=args.dataset_dir,
        dataset_rare_dir=args.dataset_rare_dir,
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