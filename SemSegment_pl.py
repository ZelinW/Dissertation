import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from ACDC_data import get_data_dir, PrepareACDC
from ResUNetpp import ResNestedUNet
from UNet_pl import UNet
from Unetpp_pl import NestedUNet
from config import get_cfg_defaults


def diceCoeff(pred, gt, smooth=1):
    activation_fn = nn.Sigmoid()
    pred = activation_fn(pred)

    N = gt.size(0)
    pred_flat = pred.view(N, -1)
    gt_flat = gt.view(N, -1)

    intersection = (pred_flat * gt_flat).sum(1)
    unionset = pred_flat.sum(1) + gt_flat.sum(1)
    loss = 2 * (intersection + smooth) / (unionset + smooth)
    return loss.sum() / N


class DiceLoss(pl.LightningModule):
    def __init__(self, num_classes):
        super(DiceLoss, self).__init__()
        self.num_classes = num_classes

    def forward(self, y_pred, y_true):
        class_dice = []
        for i in range(1, self.num_classes):
            class_dice.append(diceCoeff(y_pred[:, i:i + 1, :], y_true[:, i:i + 1, :]))
        mean_dice = sum(class_dice) / len(class_dice)
        return 1 - mean_dice


class SemSegment(pl.LightningModule):
    def __init__(self, n_channels=1, n_classes=4, batch_size=1, datadir=None):
        super().__init__()
        self.cfg = get_cfg_defaults()
        self.datadir = datadir
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.n_classes = n_classes

        if self.cfg.SOLVER.NET == 'UNet':
            self.net = UNet(n_channels=self.n_channels, n_classes=self.n_classes)
        elif self.cfg.SOLVER.NET == 'UNetpp':
            self.net = NestedUNet(n_channels=self.n_channels, n_classes=self.n_classes)
        elif self.cfg.SOLVER.NET == 'ResUNetpp':
            self.net = ResNestedUNet(n_channels=self.n_channels, n_classes=self.n_classes)

    def forward(self, x):
        if self.cfg.SOLVER.DEEPSUPERVISION and self.cfg.SOLVER.NET == 'UNetpp':
            self.net.deepsupervision = True
        return self.net(x)

    def training_step(self, batch, batch_nb):
        x, y = batch['image'], batch['label']
        y_hat = self.forward(x)
        y = y.to(dtype=torch.float32)
        lf = nn.BCEWithLogitsLoss()
        loss = lf(y_hat, y)
        dl = DiceLoss(num_classes=self.n_classes)
        dice_loss = dl(y_hat, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_DiceLoss', dice_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return dice_loss

    def validation_step(self, batch, batch_nb):
        x, y = batch['image'], batch['label']
        y_hat = self.forward(x)
        y = y.to(dtype=torch.float32)
        lf = nn.BCEWithLogitsLoss()
        loss = lf(y_hat, y)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('avg_loss', avg_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return avg_loss

    def configure_optimizers(self):
        # self.log('Learning Rate', , on_step=False, on_epoch=False, prog_bar=True, logger=True)
        return torch.optim.RMSprop(self.parameters(), lr=self.cfg.SOLVER.LR, weight_decay=1e-8, momentum=0.9)

    def __dataloader(self):
        datadir = self.datadir
        image, label = get_data_dir(datadir)
        dataset = PrepareACDC(label, image)
        n_val = int(len(dataset) * 0.1)
        n_train = len(dataset) - n_val
        train_ds, val_ds = random_split(dataset, [n_train, n_val])
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, pin_memory=True, shuffle=True)
        val_loader = DataLoader(val_ds, batch_size=self.batch_size, pin_memory=True, shuffle=False)

        return {
            'train': train_loader,
            'val': val_loader,
        }

    def train_dataloader(self):
        return self.__dataloader()['train']

    def val_dataloader(self):
        return self.__dataloader()['val']
