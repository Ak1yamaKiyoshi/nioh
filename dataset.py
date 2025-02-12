from torch.utils.data import Dataset
import pandas as pd 
from config import Config as cfg
import cv2 as cv 
import os 
import torch 
import numpy as np


class InsaneDataset(Dataset):
    def __init__(self):
        datasets = [
            "indoor_1",
            "indoor_2", 
            "indoor_3",
            "outdoor_1",
            "transition_1", 
            "transition_2",
            "transition_3",
        ]

        self._all_altitudes = []
        self._all_images = []

        for d in datasets:
            dataset_foldername = os.path.join(cfg.prefix, d)
            camera = dataset_foldername + "_nav_cam"
            sensors = dataset_foldername + "_sensors" 
            csv_images = os.path.join(camera, "nav_cam_timestamps.csv")
            csv_lrf = os.path.join(sensors, "lrf_range.csv")

            if all([os.path.exists(p) for p in [camera, sensors, csv_images, csv_lrf]]):
                df = pd.read_csv(csv_images)
                timestamps = df.to_numpy()
                
                df = pd.read_csv(csv_lrf)
                altitude = df[[cfg.col_t, cfg.col_lrf_range]].to_numpy()

                timestamps = np.array([
                    ts_row for ts_row in timestamps 
                    if min(altitude, key=lambda x: abs(x[0] - ts_row[1]))[1] >= 0.5
                ])
                
                images = []
                for idx, row in enumerate(timestamps):
                    _, stamp, name = row
                    name = f"img/{int(name)}.png"
                    image_path = os.path.join(camera, name)
                    images.append([idx, image_path, stamp])

                self._all_altitudes.extend(altitude)
                self._all_images.extend(images)

        self._all_altitudes = np.array(self._all_altitudes)
        self._all_images = np.array(self._all_images)

    def __len__(self):
        return len(self._all_images)

    def __getitem__(self, idx):
        _, image_path, stamp = self._all_images[idx]
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        stamp = float(stamp)
        alt = min(self._all_altitudes, key=lambda x: abs(x[0] - stamp))[1]
        
        img = cv.resize(img, (96, 96))
        img = torch.from_numpy(img).float() / 255.0
        img = img.view(1,96, 96)
        return img, alt


class InsaneDatasetDefaultConfig:
    datasets = [
            "indoor_1",
            "indoor_2", 
            "indoor_3",
            "outdoor_1",
            "transition_1", 
            "transition_2",
            "transition_3",
        ]


class InsaneDatasetV2(Dataset):
    def __init__(self, datasets=InsaneDatasetDefaultConfig.datasets):
        self._all_data = []

        for d in datasets:
            print(f"loading dataset... {d}")
            dataset_foldername = os.path.join(cfg.prefix, d)
            camera = dataset_foldername + "_nav_cam"
            sensors = dataset_foldername + "_sensors" 
            csv_images = os.path.join(camera, "nav_cam_timestamps.csv")
            csv_lrf = os.path.join(sensors, "lrf_range.csv")

            if all([os.path.exists(p) for p in [camera, sensors, csv_images, csv_lrf]]):
                df = pd.read_csv(csv_images)
                timestamps = df.to_numpy()
                
                df = pd.read_csv(csv_lrf)
                altitude = df[[cfg.col_t, cfg.col_lrf_range]].astype(float).to_numpy()

                timestamps = np.array([
                    ts_row for ts_row in timestamps 
                    if min(altitude, key=lambda x: abs(x[0] - ts_row[1]))[1] >= 0.5
                ])
                
                for _, stamp, name in timestamps:
                    name = f"img/{int(name)}.png"
                    image_path = os.path.join(camera, name)
                    alt = min(altitude, key=lambda x: abs(x[0] - stamp))[1]
                    self._all_data.append([image_path, float(alt)])

        self._all_data = np.array(self._all_data)

    def __len__(self):
        return len(self._all_data)

    def __getitem__(self, idx):
        image_path, alt = self._all_data[idx]
        img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
        img = cv.resize(img, (96, 96))
        img = torch.from_numpy(img).float() / 255.0
        img = img.view(1, 96, 96)
        alt = float(alt)
        return [img, alt]