import torch
import time
from utils import SimpleCNN, get_dataloader
import torch.nn as nn
import torch.optim as optim


# -----------------------------------
# Device Selection
# -----------------------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using compute device:", device)


# -----------------------------------
# Model + Data
# -----------------------------------
model = SimpleCNN().to(device)

# IMPORTANT: batch_size must match baseline tests
loader = get_dataloader(batch_size=256)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


# -----------------------------------
# Heterogeneous Pipelined Training
# -----------------------------------
start_total = time.time()

epochs = 2

for epoch in range(epochs):

    loader_iter = iter(loader)

    # ---- Load first batch ----
    images, labels = next(loader_iter)

    # Simulated CPU preprocessing
    images = images * 1.01

    # Transfer to GPU (non-blocking)
    images = images.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)

    for next_images, next_labels in loader_iter:

        # ---- CPU prepares next batch ----
        next_images = next_images * 1.01

        # ---- GPU computes current batch ----
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # ---- Transfer next batch while GPU just finished ----
        images = next_images.to(device, non_blocking=True)
        labels = next_labels.to(device, non_blocking=True)


end_total = time.time()

print("\n==============================")
print("Mode: Heterogeneous (Pipelined)")
print("------------------------------")
print("Total Time:", end_total - start_total)
print("==============================")