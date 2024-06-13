#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from mx import mx_mapping, finalize_mx_specs


# In[2]:


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


# In[3]:


def train(model, train_loader, device, epochs=10, model_path='best_model_fp32.pth'):
    if os.path.exists(model_path):
        print(f"Model file {model_path} already exists. Skipping training.")
        return model_path

    model.train()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_loss = float('inf')
    
    for epoch in range(epochs):
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

        epoch_loss /= len(train_loader)
        print(f"Epoch {epoch+1} done. Loss: {epoch_loss}")

        # Save the best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), model_path)
    
    return model_path


# In[4]:


# Data preprocessing and loading
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_data = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(root='./data', train=False, transform=transform)
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)


# In[5]:


# Evaluation function
def evaluate(model, test_loader, device):
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = (100.0 * correct) / total
    return accuracy


# In[6]:


# Model wrapper class
class ModelWrapper:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.using_fp4 = True

    def to(self, device):
        self.device = device

    def eval(self):
        self.model.eval()

    def __call__(self, x):
        return self.model(x)

    def restore_full_precision(self):
        self.inject_fp8()
        self.model.to(self.device)
        self.using_fp4 = False
        print("Switched to full precision (FP8).")

    def quantize(self):
        self.inject_fp4()
        self.model.to(self.device)
        self.using_fp4 = True
        print("Switched to quantized model (FP4).")

    def inject_fp4(self):
        mx_specs_fp4 = {
            'w_elem_format': 'fp4_e2m1',
            'a_elem_format': 'fp4_e2m1',
            'scale_bits': 8,
            'block_size': 32,
            'custom_cuda': True,
            'bfloat': 16,
        }
        mx_specs_fp4 = finalize_mx_specs(mx_specs_fp4)
        mx_mapping.inject_pyt_ops(mx_specs_fp4)

    def inject_fp8(self):
        mx_specs_fp8 = {
            'w_elem_format': 'fp8_e5m2',
            'a_elem_format': 'fp8_e5m2',
            'scale_bits': 8,
            'block_size': 32,
            'custom_cuda': True,
            'bfloat': 16,
        }
        mx_specs_fp8 = finalize_mx_specs(mx_specs_fp8)
        mx_mapping.inject_pyt_ops(mx_specs_fp8)


# In[13]:


# Anomaly detection and model swapping function
def anomaly_detection_swap(model_wrapper, test_loader, device, n):
    model_wrapper.to(device)
    model_wrapper.eval()

    consecutive_errors = 0
    consecutive_corrects = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model_wrapper(images)
            _, predicted = torch.max(outputs, 1)
            for idx in range(images.size(0)):
                if predicted[idx] == labels[idx]:
                    consecutive_errors = 0
                    consecutive_corrects += 1
                    correct += 1
                else:
                    consecutive_corrects = 0
                    consecutive_errors += 1

                # Check if we need to switch model precision
                if model_wrapper.using_fp4 and consecutive_errors > n:
                    model_wrapper.restore_full_precision()
                    consecutive_errors = 0  # Reset counter after switching
                    accuracy = evaluate(model_wrapper.model, test_loader, device)
                    print(f"Evaluation after switching to FP8: {accuracy:.6f}%")
                elif not model_wrapper.using_fp4 and consecutive_corrects > n:
                    model_wrapper.quantize()
                    consecutive_corrects = 0  # Reset counter after switching
                    accuracy = evaluate(model_wrapper.model, test_loader, device)
                    print(f"Evaluation after switching to FP4: {accuracy:.6f}%")

                total += 1

    accuracy = (correct / total) * 100
    print(f"Final accuracy: {accuracy:.6f}%. Final swap lead to {'FP4' if model_wrapper.using_fp4 else 'FP8'} model")


# In[8]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# In[9]:


# Train the model in FP32 for 10 epochs
model_fp32 = SimpleCNN().to(device)
best_model_fp32_path = train(model_fp32, train_loader, device, epochs=10, model_path='best_model_fp32.pth')
model_fp32.load_state_dict(torch.load(best_model_fp32_path))


# In[10]:


# Evaluate the trained FP32 model
accuracy_fp32 = evaluate(model_fp32, test_loader, device)
print(f"FP32 - Accuracy: {accuracy_fp32:.6f}%")


# In[11]:


# Create a model wrapper
model_wrapper = ModelWrapper(model_fp32, device)


# In[14]:


# Anomaly detection and model swapping
anomaly_detection_swap(model_wrapper, test_loader, device, n=5)

