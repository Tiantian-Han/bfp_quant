import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 8 * 8)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train(model, train_loader, device, model_path='best_model_fp32.pth'):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_loss = float('inf')
    epoch = 0

    try:
        while True:
            epoch_loss = 0
            for data, target in train_loader:
                data = data.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = F.cross_entropy(output, target)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

                # Save the model state after every batch
                torch.save(model.state_dict(), 'latest_model.pth')

            epoch_loss /= len(train_loader)
            epoch += 1
            print(f"Epoch {epoch} done. Loss: {epoch_loss}")

            # Save the best model
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                torch.save(model.state_dict(), model_path)
    except KeyboardInterrupt:
        print(f"Training interrupted at epoch {epoch}. Saving latest state...")

    return model_path

# Data preprocessing and loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Train the model in FP32 for an indefinite number of epochs
model_fp32 = SimpleCNN().to(device)
best_model_fp32_path = train(model_fp32, train_loader, device, model_path='best_model_fp32.pth')
