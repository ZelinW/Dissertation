import torch

from ACDC_data import get_data_dir, PrepareACDC
from config import get_cfg_defaults
from train_model import train_net
from model import Unetpp_pl
from model import UNet_pl


def main():
    cfg = get_cfg_defaults()
    cfg.freeze()
    # ---- dataset ----
    image, label = get_data_dir(cfg.DATASET.PATH)
    data = PrepareACDC(label, image)

    # ---- set model ----
    net = UNet_pl.UNet(n_channels=1, n_classes=4, bilinear=True)
    # net = Unetpp.NestedUNet(n_channels=1, n_classes=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    net = net.to(device=device)
    train_net(data,
              net=net,
              epochs=1000,
              batch_size=1,
              lr=0.0001,
              device=device,
              val_percent=0.1)


if __name__ == '__main__':
    main()
