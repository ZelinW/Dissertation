import os
import random

import cv2
import nibabel as nib
import numpy as np
from scipy import ndimage
from torch.utils.data import Dataset


def get_data_dir(input_folder):
    data_file = list()
    data_label = list()
    for folder in os.listdir(input_folder):
        folder_path = os.path.join(input_folder, folder)
        if os.path.isdir(folder_path):
            infos = {}
            for line in open(os.path.join(folder_path, 'Info.cfg')):
                label, value = line.split(':')
                if label == 'ED' or label == "ES":
                    infos[label] = value.rstrip('\n').lstrip(' ').rjust(2, '0')
            patient = folder_path.rsplit(os.sep)[-1]
            for key, value in infos.items():
                data_file.append(os.path.join(folder_path, "{}_frame{}.nii.gz".format(patient, value)))
                data_label.append(os.path.join(folder_path, "{}_frame{}_gt.nii.gz".format(patient, value)))
    return data_file, data_label


def load_nii(img_path):
    nimg = nib.load(img_path)
    return nimg.get_data()


class PrepareACDC(Dataset):
    def __init__(self, labels, images, scale=0.5):
        self.labels = labels
        self.images = images
        self.scale = scale

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = load_nii(self.images[idx])
        label = load_nii(self.labels[idx])
        image = crop_image(image)
        x, y, z = image.shape
        t = random.randint(0, z - 1)
        p = np.expand_dims(crop_image(label)[:, :, t], axis=0)
        image = np.expand_dims(rescale_intensity(image)[:, :, t], axis=0)
        image, d = data_augmenter(image, p, shift=10, rotate=10, scale=0.1,
                                  intensity=0.1, flip=False)
        f = mask2onehot(np.squeeze(p), 4)

        sample = {"image": image, "label": f}
        return sample


def mask2onehot(mask, num_classes):
    """
    Converts a segmentation mask (H,W) to (K,H,W) where the last dim is a one
    hot encoding vector
    """
    # _mask = [mask == i for i in range(num_classes)]
    _mask = []
    for i in range(num_classes):
        if i == 0:
            _mask.append(np.zeros((mask.shape[:2])))
        else:
            t = mask == i
            _mask.append(t * i)
    return np.array(_mask).astype(np.uint8)


def onehot2mask(mask):
    """
    Converts a mask (K, H, W) to (H,W)
    """
    _mask = np.argmax(mask, axis=0).astype(np.uint8)
    return _mask


def rescale_intensity(image, thres=(1.0, 99.0)):
    """ Rescale the image intensity to the range of [0, 1] """
    val_l, val_h = np.percentile(image, thres)
    image2 = image
    image2[image < val_l] = val_l
    image2[image > val_h] = val_h
    image2 = (image2.astype(np.float32) - val_l) / (val_h - val_l)
    return image2


def crop_image(image, size=192):
    """ Crop a 3D image using a bounding box centred at (cx, cy) with specified size """
    X, Y, Z = image.shape
    cx, cy = int(X / 2), int(Y / 2)

    r = int(size / 2)
    x1, x2 = cx - r, cx + r
    y1, y2 = cy - r, cy + r
    x1_, x2_ = max(x1, 0), min(x2, X)
    y1_, y2_ = max(y1, 0), min(y2, Y)
    # Crop the image
    crop = image[x1_: x2_, y1_: y2_]
    # Pad the image if the specified size is larger than the input image size
    if crop.ndim == 3:
        crop = np.pad(crop,
                      ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (0, 0)),
                      'constant')
    elif crop.ndim == 4:
        crop = np.pad(crop,
                      ((x1_ - x1, x2 - x2_), (y1_ - y1, y2 - y2_), (0, 0), (0, 0)),
                      'constant')
    else:
        print('Error: unsupported dimension, crop.ndim = {0}.'.format(crop.ndim))
        exit(0)
    return crop


def data_augmenter(image, label, shift=0.0, rotate=0.0, scale=0.0, intensity=0.0, flip=False):
    image2 = np.zeros(image.shape, dtype=np.float32)
    label2 = np.zeros(label.shape, dtype=np.int32)
    for i in range(image.shape[0]):
        # For each image slice, generate random affine transformation parameters
        # using the Gaussian distribution
        shift_val = [np.clip(np.random.normal(), -3, 3) * shift,
                     np.clip(np.random.normal(), -3, 3) * shift]
        rotate_val = np.clip(np.random.normal(), -3, 3) * rotate
        scale_val = 1 + np.clip(np.random.normal(), -3, 3) * scale
        intensity_val = 1 + np.clip(np.random.normal(), -3, 3) * intensity

        # Apply the affine transformation (rotation + scale + shift) to the image
        row, col = image.shape[1:3]
        M = cv2.getRotationMatrix2D((row / 2, col / 2), rotate_val, 1.0 / scale_val)
        M[:, 2] += shift_val
        image2[i, :, :] = ndimage.interpolation.affine_transform(image[i, :, :],
                                                                 M[:, :2], M[:, 2], order=1)

        # Apply the affine transformation (rotation + scale + shift) to the label map
        label2[i, :, :] = ndimage.interpolation.affine_transform(label[i, :, :],
                                                                 M[:, :2], M[:, 2], order=0)

        # Apply intensity variation
        image2[i] *= intensity_val

        # Apply random horizontal or vertical flipping
        if flip:
            if np.random.uniform() >= 0.5:
                image2[i] = image2[i, ::-1, :]
                label2[i] = label2[i, ::-1, :]
            else:
                image2[i] = image2[i, :, ::-1]
                label2[i] = label2[i, :, ::-1]
    return image2, label2