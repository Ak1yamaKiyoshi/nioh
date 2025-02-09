from config import Config as cfg

import os, sys
import psutil

import pandas as pd
import numpy as np 
import cv2 as cv

import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision.transforms import transforms
from torchvision.models.inception import Inception3



def get_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    memory_mb = mem_info.rss / 1024 / 1024
    return memory_mb


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


class Block(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.conv3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels, out_channels, kernel_size=5, stride=1, padding=2) 
        self.conv7 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=1, padding=3)

        self.selector = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 3, 1),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        weights = self.selector(x)
        conv3_out = self.conv3(x)
        conv5_out = self.conv5(x)
        conv7_out = self.conv7(x)

        out = (weights[:, 0:1] * conv3_out + 
               weights[:, 1:2] * conv5_out + 
               weights[:, 2:3] * conv7_out)
        return out


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        hidden = 100
        hidden_out = 32
        self.conv2fc = 512
        self.first_run = True
        self.conv = nn.Sequential(
            Block(1, hidden),
            nn.BatchNorm2d(hidden),
            nn.ReLU(), 
            Block(hidden, hidden_out),
            nn.BatchNorm2d(hidden_out),
            nn.ReLU()
        )

        self.conv_out = nn.Sequential(
            nn.Conv2d(hidden_out, hidden_out*2, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_out*2, 32, 3, stride=5, padding=1),
            nn.ReLU(),
        )

        self.block_fc = nn.Sequential(
            nn.Linear(32*10*10, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.ReLU(),
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        x = x.view(x.size(0), -1)

# img, alt = dat[1600]
# 
# cv.imshow(f"{alt:3.2f}", img)
# cv.waitKey(0)
# 
# #plt.plot(altitude_data[:, 1])
# #plt.show() 
# 
# print(df.to_numpy())

        x = self.block_fc(x)
        return x.view(-1)

class RMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, pred, actual):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(actual + 1)))


dataset = InsaneDataset()
loader = DataLoader(dataset, 8, True)


model = Model()
criterion = RMSLELoss()
device = cuda = torch.device("cuda")
model.to(cuda)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05,)
losses = []

print('training...?')
for epoch in range(100):
    epoch_loss = 0
    for input in loader:
        img = input[0].to(cuda, dtype=torch.float32)
        alt = input[1].to(cuda, dtype=torch.float32)
        out = model(img)
        logits = model(img)
        loss = criterion(logits.cpu(), alt.cpu())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        break
    
    img, alt = next(iter(loader))
    img = input[0].to(cuda, dtype=torch.float32)
    alt = input[1].to(cuda, dtype=torch.float32)
    pred = model(img)
    print(f"pred: {pred}, alt: {alt}")
    print(epoch_loss)


# TODO: 
# - Cutoff from dataset alt < 0.5meters. 
# - evaluation dataset 
# - print(f"\r batch: {i+1}/{len(train_dataloader)}, batch/s: {avg_batch_time:3.2f}, remaining: {time_remaining_s:5.1f}s., batch loss: {loss.item():12.5f}, epoch loss: {epoch_loss:12.5f}, lr: {learning_rate}", end="", flush=False)
# - checkpoint wiritn'
