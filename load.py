from model import Model, load_checkpoint
from dataset import InsaneDatasetV2
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

dataset = InsaneDatasetV2(
    ["indoor_1"]
)
print(len(dataset))
loader = DataLoader(dataset, 1, False)

model, epoch, loss = load_checkpoint("checkpoints/cp_[7]_Model_v4_Long_ConvFC_(2025.02.11-16:54:09)_(l:47.63).pth")
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
