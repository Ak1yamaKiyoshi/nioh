from model import Model, load_checkpoint
from dataset import InsaneDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

dataset = InsaneDataset()
loader = DataLoader(dataset, 1, False)

model, epoch, loss = load_checkpoint("checkpoints/cp_[82]_Model_v1_(2025.02.10-08:02:18)_(l:6.52).pth")
model.eval()

predictions = []
actual = []

i = 0 
for sample in loader:
    img = sample[0]
    real = sample[1]
    pred = model(img).cpu().detach().numpy()[0]
    real = real.cpu().detach().numpy()[0]
    predictions.append(pred)
    actual.append(real)
    i+=1
    
    print(f"\r {i:4d}/{len(loader)}", end="", flush=False)

plt.plot(actual, label="actual")
plt.plot(predictions, label="predictions")
plt.legend()
plt.show()


#print(model)
