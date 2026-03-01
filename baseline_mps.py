import torch
from utils import SimpleCNN, get_dataloader, train

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    print("MPS not available, falling back to CPU")
    device = torch.device("cpu")

print("Using device:", device)

model = SimpleCNN()
loader = get_dataloader(batch_size=256)

time_taken = train(model, device, loader, epochs=2)

print("Training Time (MPS):", time_taken)