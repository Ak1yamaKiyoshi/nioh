
from torch import nn 
import torch 
from datetime import datetime




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


class Model_v2(nn.Module):
    def __init__(self):
        super().__init__()
        hidden = 32
        hidden_out = 16
        self.conv2fc = 512
        self.first_run = True
        self.conv = nn.Sequential(
            Block(1, 96),
            nn.BatchNorm2d(96),
            nn.ReLU(), 
            Block(96, hidden),
            nn.BatchNorm2d(hidden),
            nn.ReLU(), 
            Block(hidden, hidden),
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
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.ReLU(),
        )

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        x = x.view(x.size(0), -1)

        x = self.block_fc(x)
        return x.view(-1)

Model = Model_v2


def checkpoint_name(epoch, model_name, epoch_loss) -> str:
    datestr = datetime.now().strftime("(%Y.%m.%d-%H:%M:%S)")
    return f"cp_[{epoch}]_{model_name}_{datestr}_(l:{epoch_loss:3.2f}).pth"

def save_checkpoint(model, optimizer, epoch, loss):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model': model,
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'loss': loss
    }
    
    path = f'checkpoints/{checkpoint_name(epoch, f"{model.__class__.__name__}", loss)}'
    torch.save(checkpoint, path)
    return path

def load_checkpoint(path, model=None, optimizer=None):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    
    if model is None:
        model = checkpoint['model']
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
        
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
    return model, checkpoint['epoch'], checkpoint.get('loss', None)