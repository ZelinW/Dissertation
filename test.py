import matplotlib.pyplot as plt
import numpy as np
import torch

from ACDC_data import load_nii, crop_image, rescale_intensity, data_augmenter, onehot2mask
from model.SemSegment_pl import SemSegment
from config import get_cfg_defaults


def accuracy(pred, truth):
    return np.mean(np.equal(pred, truth))


cfg = get_cfg_defaults()

# model = SemSegment.load_from_checkpoint(
#     "D:\PROGRAM\GITHUB\ACDC_UNet\checkpoints_UNetpp\last.ckpt")
# model.deepsupervision = True

image = load_nii('D:\PROGRAM\GITHUB\ACDC_UNet\ADCD_data\patient001\patient001_frame01.nii.gz')
label = load_nii('D:\PROGRAM\GITHUB\ACDC_UNet\ADCD_data\patient001\patient001_frame01_gt.nii.gz')
image = crop_image(image)
x, y, z = image.shape

label = np.expand_dims(crop_image(label)[:, :, 3], axis=0)
image = np.expand_dims(rescale_intensity(image)[:, :, 3], axis=0)
# image = np.expand_dims(image[:, :, t], axis=0)
image, t = data_augmenter(image, label)
image = np.expand_dims(image, 0)

pred = SemSegment.load_from_checkpoint(
    "D:\PROGRAM\GITHUB\ACDC_UNet\checkpoints_UNetpp\last.ckpt").forward(torch.tensor(image))

# pred = model(torch.tensor(image))
t = torch.squeeze(pred).detach().numpy()
target = onehot2mask(t)

acc = accuracy(label, target)

plt.figure()
if cfg.SOLVER.NET == 'UNet':
    plt.suptitle('UNet accuracy: {:.2%}'.format(acc), size=25)
else:
    plt.suptitle('UNet++ accuracy: {:.2%}'.format(acc), size=25)

plt.subplot(1, 2, 1)
plt.imshow(target)
plt.title("Predict", size=20)

plt.subplot(1, 2, 2)
plt.imshow(np.squeeze(label))
plt.title("Label", size=20)
plt.show()
