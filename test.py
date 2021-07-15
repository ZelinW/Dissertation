import matplotlib.pyplot as plt
import numpy as np
import torch

from ACDC_data import load_nii, crop_image, rescale_intensity, data_augmenter, onehot2mask, mask2onehot
from config import get_cfg_defaults
from model.SemSegment_pl import SemSegment


def accuracy(pred, truth):
    return np.mean(np.equal(pred, truth))


cfg = get_cfg_defaults()

image = load_nii('D:\PROGRAM\GITHUB\ACDC_UNet\ACDC_data\patient001\patient001_frame01.nii.gz')
label = load_nii('D:\PROGRAM\GITHUB\ACDC_UNet\ACDC_data\patient001\patient001_frame01_gt.nii.gz')
image = crop_image(image)
x, y, z = image.shape

label = np.expand_dims(crop_image(label)[:, :, 3], axis=0)
image = np.expand_dims(rescale_intensity(image)[:, :, 3], axis=0)

image, t = data_augmenter(image, label)
image = np.expand_dims(image, 0)

pred = SemSegment.load_from_checkpoint(
    "D:\PROGRAM\GITHUB\ACDC_UNet\checkpoints\checkpoints_UNetpp\last.ckpt").forward(torch.tensor(image))

if cfg.SOLVER.DEEPSUPERVISION:
    label = mask2onehot(np.squeeze(label), num_classes=4)
    pred = pred[3]  # 第几层
    pred = torch.squeeze(pred).detach().numpy()
    acc = accuracy(label[3], pred[3])

    plt.figure()
    if cfg.SOLVER.NET == 'UNet':
        plt.suptitle('UNet accuracy: {:.2%}'.format(acc), size=25)
    else:
        plt.suptitle('UNet++ accuracy: {:.2%}'.format(acc), size=25)

    plt.subplot(1, 2, 1)
    plt.imshow(pred[3])
    plt.title("Predict", size=20)

    plt.subplot(1, 2, 2)
    plt.imshow(np.squeeze(label)[3])
    plt.title("Label", size=20)
    plt.show()

else:
    pred = torch.squeeze(pred).detach().numpy()
    pred = onehot2mask(pred)
    acc = accuracy(label, pred)
    plt.figure()
    if cfg.SOLVER.NET == 'UNet':
        plt.suptitle('UNet accuracy: {:.2%}'.format(acc), size=25)
    else:
        plt.suptitle('UNet++ accuracy: {:.2%}'.format(acc), size=25)

    plt.subplot(1, 2, 1)
    plt.imshow(pred)
    plt.title("Predict", size=20)

    plt.subplot(1, 2, 2)
    plt.imshow(np.squeeze(label))
    plt.title("Label", size=20)
    plt.show()
