from config import Config as cfg

import os, sys
import psutil

import pandas as pd
import numpy as np 
import cv2 as cv

import time
from datetime import datetime

import matplotlib
matplotlib.use('qtagg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import transforms
from torchvision.models.inception import Inception3
from model import Model, save_checkpoint, load_checkpoint

from dataset import InsaneDataset
from loss import RMSLELoss


def get_memory_usage():
    process = psutil.Process()
    mem_info = process.memory_info()
    memory_mb = mem_info.rss / 1024 / 1024
    return memory_mb


def print_model_parameters(model):
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            params = param.numel()
            print(f"{name}: {params:,} parameters")
            total_params += params
    print(f"\nTotal trainable parameters: {total_params:,}")


def prettify_float(l):
    output = ""
    for e in l:
        output += f"{e:5.1f}, "
    return output

print("Loading dataset")
dataset = InsaneDataset()
print("Dataset loaded")


loader = DataLoader(dataset, 64, True)

model = Model()
model, start_epoch, loss = load_checkpoint("checkpoints/cp_[83]_Model_v1_(2025.02.10-19:58:23)_(l:30.43).pth")
print_model_parameters(model)


criterion = RMSLELoss()
device = cuda = torch.device("cuda")
model.to(cuda)
optimizer = torch.optim.Adam(model.parameters(), lr=0.05,)
losses = []

total_epochs_train = 150


for epoch in range(total_epochs_train):
    epoch_loss = 0
    epoch = start_epoch + epoch

    # Train pass
    batches_in_loader = len(loader)
    times_per_batch = []

    for i, batch in enumerate(loader):
        time_batch_start = time.monotonic()

        img = batch[0].to(cuda, dtype=torch.float32)
        alt = batch[1].to(cuda, dtype=torch.float32)
        logits = model(img)

        loss = criterion(logits.cpu(), alt.cpu())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        time_batch_end = time.monotonic()
        time_per_batch = time_batch_end - time_batch_start
        times_per_batch.append(time_per_batch)
        mean_time_batch = np.mean(times_per_batch)
        remained_time_estimate = (batches_in_loader - i+1) * mean_time_batch

        loss_per_batch = loss.item()
        epoch_loss += loss_per_batch
        print(f"\r[{i+1:4d}/{batches_in_loader:4d}]  bloss: {loss_per_batch:5.2f}, complete in: {remained_time_estimate:7.2f}s.", end="", flush=False)
    print()
    
    img, alt = next(iter(loader))
    img = batch[0].to(cuda, dtype=torch.float32)
    alt = batch[1]
    pred = model(img)
    
    actual_altitude = alt.detach().numpy().tolist()
    predicted_altitude = pred.cpu().detach().numpy().tolist()

    print(f""" 
[ - -- - ] Epoch: {epoch+1:4d}/{total_epochs_train:4d}
[ - -- - ] Loss: {epoch_loss}
[ - -- - ] actual: {prettify_float(actual_altitude)}
[ - -- - ] pred:   {prettify_float(predicted_altitude)}
""")

    if not os.path.exists('checkpoints/'):
        os.mkdir('checkpoints')

    save_checkpoint(model, optimizer, epoch, epoch_loss)
    torch.cuda.empty_cache()
    


# TODO: 
# - evaluation dataset
# - rewrite dataloader, make sure that in one batch there is 0-1, 15-20, 1-15 alts in same amount 