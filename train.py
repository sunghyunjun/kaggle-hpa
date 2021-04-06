from argparse import ArgumentParser
import os

import pytorch_lightning as pl
from pytorch_lightning.loggers import NeptuneLogger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint

def main():
    # ----------
    # seed
    # ----------
    pl.seed_everything(0)

    # ----------
    # args
    # ----------
    parser = ArgumentParser()
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()

    # ----------
    # for debug
    # ----------
    if args.debug:
        args.max_epochs = 1
        args.limit_train_batches = 3
        args.limit_val_batches = 3

    # ----------
    # data
    # ----------

    # ----------
    # model
    # ----------

    # checkpoint_callback = ModelCheckpoint(
    #     monitor="val_loss",
    #     filename="xray-detector-{epoch:03d}-{val_loss:.4f}",
    #     save_last=True,
    #     save_top_k=3,
    #     mode="min",
    # )

    # ----------
    # logger
    # ----------
    # if args.neptune_logger:
    #     logger = NeptuneLogger(
    #         api_key=os.environ["NEPTUNE_API_TOKEN"],
    #         project_name=args.neptune_project,
    #         experiment_name=args.experiment_name,
    #         params=args.__dict__,
    #         tags=["pytorch-lightning"],
    #     )
    # else:
    #     # default logger: TenserBoardLogger
    #     logger = True

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # ----------
    # training
    # ----------
    trainer = pl.Trainer.from_argparse_args(
        args, callbacks=[checkpoint_callback, lr_monitor], logger=logger
    )

if __name__ == "__main__":
    main()