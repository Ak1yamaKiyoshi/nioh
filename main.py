from torch.utils.data import DataLoader, Dataset
from torch import nn
import torch
import numpy as np
import os 
import cv2
from loss import RMSLELoss
from torch.optim.adam import Adam


def parse(filename: str):
    to_human = {
        "r": "roll_degrees",
        "p": "pitch_degrees",
        "y": "yaw_degrees",
        "R": "roll_rate_degrees",
        "P": "pitch_rate_degrees",
        "Y": "yaw_rate_degrees",
        "x": "accelerometer_x_meters",
        "y": "accelerometer_y_meters",
        "v": "accelerometer_z_meters",
        "z": "altitude_meters",
    }

    data_part = filename.split('_', 2)[-1]

    result = {}
    pos = 0
    while pos < len(data_part):
        key = data_part[pos]
        pos += 1
        next_pos = pos
        while next_pos < len(data_part) and not data_part[next_pos].isalpha():
            next_pos += 1
        value = float(data_part[pos:next_pos])
        result[to_human[key]] = value
        pos = next_pos
    return result


class OwnDataset(Dataset): 
    def __init__(self, path:str):
        super().__init__()
        self._dataset_path = path
        self._samples_list = os.listdir(path)

    def __getitem__(self, index):
        filename = self._samples_list[index]
        data = parse(filename.replace(".png", ""))
        image_path = os.path.join(self._dataset_path, filename)
        image = cv2.imread(image_path)
        if image is None:
            print(image_path)

        image = np.transpose(image.astype(np.float32) / 255, (2, 0, 1)) 

        return {
            "x": image,
            "y": data['altitude_meters']
        }


    def __len__(self):
        return len(self._samples_list)

    
class I2HModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.enc3 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.center = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1)
        )
        
        # Final prediction
        self.predict = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.ReLU()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        
        c = self.center(e3)   
        c = c.view(c.size(0), -1) 

        out = self.predict(c)
        return out

dataset = OwnDataset("owndataset")
loader = DataLoader(dataset, batch_size=16, shuffle=True)
criterion = RMSLELoss()

model = I2HModel()
model = model.to("cuda")
epochs=  20

optimizer = Adam(model.parameters())


for epoch in range(epochs):
    epoch_loss = 0
    for i, sample in enumerate(loader):
        y = sample['y'].float()
        logits = model(sample['x'].to('cuda').float()).flatten()
        loss = criterion(y, logits.to("cpu"))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        print(f"\r [{i}/{len(loader)}]", end="", flush=False)
    
    print(f"{epoch+1}/{epochs}: {epoch_loss}")