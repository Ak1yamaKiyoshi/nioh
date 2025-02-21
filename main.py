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

from dataset import InsaneDatasetV2
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

print("Loading Training dataset")
dataset = InsaneDatasetV2([
    "outdoor_1",
    "transition_1",
    "transition_3",
    "indoor_2", 
    "indoor_1",
])

print("Loading Validation dataset ")
validation_dataset = InsaneDatasetV2([
    "transition_2", 
    "indoor_3",
])

loader = DataLoader(dataset, 16, True)
validation_loader = DataLoader(validation_dataset, 1, False)


model = Model()
start_epoch = 0
# model, epoch, loss = load_checkpoint("checkpoints/cp_[7]_Model_v4_Long_ConvFC_(2025.02.11-16:54:09)_(l:47.63).pth")
print_model_parameters(model)


criterion = RMSLELoss()
device = cuda = torch.device("cuda")
model.to(cuda)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005,)
losses = []

total_epochs_train = 150

metadata = []
datestr = datetime.now().strftime("(%Y.%m.%d-%H:%M:%S)")


save_checkpoint(model, optimizer, 0, 0, datestr, metadata)

for epoch in range(total_epochs_train):
    epoch_loss = 0
    epoch = start_epoch + epoch

    # Train pass
    batches_in_loader = len(loader)
    times_per_batch = []

    model.train()
    time_batch_start = time.monotonic()
    for i, batch in enumerate(loader):

        try:
            img = batch[0].to(cuda, dtype=torch.float32)
            alt = batch[1].to(cuda, dtype=torch.float32)
            logits = model(img)
            

            loss = criterion(logits.cpu(), alt.cpu())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            time_batch_end = time.monotonic()
            time_per_batch = time_batch_end - time_batch_start
            time_batch_start = time.monotonic()
            times_per_batch.append(time_per_batch)
            mean_time_batch = np.mean(times_per_batch)
            remained_time_estimate = (batches_in_loader - i+1) * mean_time_batch

            loss_per_batch = loss.item()
            epoch_loss += loss_per_batch
            print(f"\r[{i+1:4d}/{batches_in_loader:4d}] (batch/{mean_time_batch:5.2f}s.)  (batch loss: {loss_per_batch:5.2f}) complete in: {remained_time_estimate:7.2f}s.", end="", flush=False)
        except BaseException as e:
            print(e)
    print("\n Validation run...")

    model.eval()
    batches_in_loader = len(validation_loader)
    times_per_batch = []
    validation_loss = 0

    for i, batch in enumerate(validation_loader):
        img = batch[0].to(cuda, dtype=torch.float32)
        alt = batch[1].to(cuda, dtype=torch.float32)
        logits = model(img)

        loss = criterion(logits.cpu(), alt.cpu())
        optimizer.zero_grad()

        time_batch_end = time.monotonic()
        time_per_batch = time_batch_end - time_batch_start
        time_batch_start = time.monotonic()
        times_per_batch.append(time_per_batch)
        mean_time_batch = np.mean(times_per_batch)
        remained_time_estimate = (batches_in_loader - i+1) * mean_time_batch

        loss_per_batch = loss.item()
        validation_loss += loss_per_batch
        print(f"\r[{i+1:4d}/{batches_in_loader:4d}] (batch/{mean_time_batch:5.2f}s.)  (batch loss: {loss_per_batch:5.2f}) complete in: {remained_time_estimate:7.2f}s.", end="", flush=False)
    print()

    img, alt = next(iter(loader))
    img = batch[0].to(cuda, dtype=torch.float32)
    alt = batch[1]
    pred = model(img)

    actual_altitude = alt.detach().numpy().tolist()
    predicted_altitude = pred.cpu().detach().numpy().tolist()
    metadata.append({
        "epoch": epoch+1,
        "train_loss": epoch_loss,
        "train_loss_per_batch": epoch_loss/len(loader),
        "validation_loss": validation_loss,
        "validation_loss_per_batch": validation_loss/len(validation_loader)
    })
    

    print(f""" 
[ - -- - ] Epoch: {epoch+1:4d}/{total_epochs_train:4d}
[ - -- - ] Train loss: {epoch_loss} 
[ - -- - ] Train loss (per batch): {epoch_loss/len(loader)} 
[ - -- - ] Validation loss: {validation_loss}
[ - -- - ] Validation loss (per batch): {validation_loss/len(validation_loader)} 
""")

    if not os.path.exists('checkpoints/'):
        os.mkdir('checkpoints')

    save_checkpoint(model, optimizer, epoch, epoch_loss, datestr, metadata)
    torch.cuda.empty_cache()


# TODO: 
# - rewrite dataloader, make sure that in one batch there is 0-1, 15-20, 1-15 alts in same amount 
# - huge seconds in time estimage to minutes and hours instead.s

""" 
    two approaches:
        - video to distance 
            [allow use of trackers, math and kalman filtering]
        - use backbone like resnet 

"""
