from model import Model, load_checkpoint

model, epoch, loss = load_checkpoint("checkpoints/cp_1_model_20250209-192601_1.47.pth")
print(model)