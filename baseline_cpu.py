import torch
from utils import SimpleCNN, get_dataloader, train
print("Starting CPU training...")

device = torch.device("cpu")
print("Using device:", device)

model = SimpleCNN()
loader = get_dataloader(batch_size=256)

time_taken = train(model, device, loader, epochs=2)

print("Training Time (CPU):", time_taken)