from torch.utils.data import Dataset
import pandas as pd 
from config import Config as cfg
import cv2 as cv 
import os 
import torch 


class InsaneDataset(Dataset):
    def __init__(self):
        self._path_image_timestamps_csv = cfg.path_image_timestamps_csv
        self._path_sensors_lrf_range_csv = cfg.path_lrf_range_csv

        self._images_folder = cfg.path_images_dataset
        self._imageprefix = os.path.join(self._images_folder, "img")

        df = pd.read_csv(self._path_image_timestamps_csv)
        self._timestamps = df.to_numpy()
        df = pd.read_csv(self._path_sensors_lrf_range_csv)
        self._altitude = df[[cfg.col_t, cfg.col_lrf_range]].to_numpy()

    def __len__(self):
        return self._timestamps.shape[0]-1

    def __getitem__(self, idx):
        _, stamp, name = self._timestamps[idx]

        name = f"{int(name)}.png"

        abs_imagepaths = os.path.join(self._imageprefix, name)

        img = cv.imread(abs_imagepaths, cv.IMREAD_GRAYSCALE)
        alt = min(self._altitude, key=lambda x: abs(x[0] - stamp))[1]

        img = cv.resize(img, (96, 96))
        img = torch.from_numpy(img).float() / 255.0
        img = img.view(1, 96, 96)
        return img, alt
