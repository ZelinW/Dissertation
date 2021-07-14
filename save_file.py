import os

from config import get_cfg_defaults

cfg = get_cfg_defaults()


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


data_file, data_label = get_data_dir(cfg.DATASET.PATH)

