import os

from pytorch_lightning import Trainer
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from model.SemSegment_pl import SemSegment
from config import get_cfg_defaults


def main():
    cfg = get_cfg_defaults()

    if cfg.SOLVER.NET == 'UNet':
        log_dir = '.\checkpoints\checkpoints_UNet'

    elif cfg.SOLVER.NET == 'UNetpp':
        if cfg.SOLVER.DEEPSUPERVISION:
            log_dir = '.\checkpoints\checkpoints_UNetpp_DS'
        else:
            log_dir = '.\checkpoints\checkpoints_UNetpp'

    elif cfg.SOLVER.NET == 'ResUNetpp':
        log_dir = '.\checkpoints\checkpoints_ResUNetpp'


    model = SemSegment(n_channels=cfg.SOLVER.N_CHANNELS,
                       n_classes=cfg.SOLVER.N_CLASSES,
                       batch_size=cfg.SOLVER.BATCH_SIZE,
                       datadir=cfg.DATASET.PATH,
                       )

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=os.getcwd(),
        version=None,
        name='lightning_logs',
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='train_loss_epoch',
        filename='sample-mnist-{epoch:02d}-{val_loss:.2f}',
        save_top_k=20,
        mode='min',
        every_n_train_steps=100,
        save_last=True,
        dirpath=log_dir,
    )

    stop_callback = EarlyStopping(
        monitor='train_loss',
        min_delta=0.0,
        patience=cfg.SOLVER.PATIENCE,
        verbose=False,
        mode='min',
        strict=True
    )

    trainer = Trainer(
        gpus=1,
        callbacks=[stop_callback, checkpoint_callback],
        logger=tb_logger,
        max_epochs=cfg.SOLVER.MAX_EPOCHS,
    )

    trainer.fit(model)


if __name__ == '__main__':
    main()
