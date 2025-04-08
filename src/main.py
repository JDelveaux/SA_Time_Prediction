# -*- coding: utf-8 -*-
"""
Loads the datasets, converts to PyTorch format, and trains/evaluates a neural network.

Author: jaked
"""

# -------------------------------
# Imports
# -------------------------------
import torch
import numpy as np
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Lambda
from torcheval.metrics import MulticlassF1Score

# -------------------------------
# Device Configuration
# -------------------------------
device = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)

# -------------------------------
# Dataset Configuration
# -------------------------------
dataset_name = "all"
conv = False
output_option = 1
weighted_output = False

# -------------------------------
# Load Data
# -------------------------------
PATH = "../data/"
data_path = f"{PATH}{dataset_name}_data_{output_option}.npy"
data = np.load(data_path, allow_pickle=True)

# -------------------------------
# Custom Dataset Class
# -------------------------------
class SAMHSADataset(Dataset):
    def __init__(self, np_array, target_transform=None):
        self.data = np_array
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        inp = torch.from_numpy(self.data[idx, 1:].astype(float))
        label = self.data[idx, 0] - 1  # Convert range 1–37 to 0–36
        label = self.target_transform(label) if self.target_transform else torch.tensor(label)
        return inp.float(), label.float()

# -------------------------------
# Weight Sampling (Optional)
# -------------------------------
class_sample_count = np.array([len(np.where(data[:, 0] == t)[0]) for t in np.unique(data[:, 0])])
weight = 1. / class_sample_count
samples_weight = torch.from_numpy(weight / weight.min()) if weighted_output else None

# -------------------------------
# Dataset Initialization
# -------------------------------
if output_option:
    target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1))
    dataset = SAMHSADataset(data, target_transform=target_transform)
else:
    dataset = SAMHSADataset(data)

# -------------------------------
# Neural Network Definition
# -------------------------------
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        if conv:
            self.conv1 = nn.Conv1d(1, 25, 5, padding=3)
            self.pool = nn.MaxPool1d(5)
            self.conv2 = nn.Conv1d(25, 107, 5, padding=3)
            self.fc1 = nn.Linear(1391, 1024)
            self.fc2 = nn.Linear(1024, 512)
            self.fc3 = nn.Linear(512, 10)
        else:
            input_dim = 329 if dataset_name == "all" else 321
            self.linear_stack = nn.Sequential(
                nn.Linear(input_dim, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 10)
            )
            self.flatten = nn.Flatten()

    def forward(self, x):
        if conv:
            x = self.pool(F.tanh(self.conv1(x)))
            x = self.pool(F.tanh(self.conv2(x)))
            x = torch.flatten(x, 1)  # Flatten all dimensions except batch
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
            return self.fc3(x)
        else:
            x = self.flatten(x)
            return self.linear_stack(x)

# -------------------------------
# Model Instantiation
# -------------------------------
model = NeuralNetwork().to(device)

# -------------------------------
# Training Loop
# -------------------------------
def train_loop(dataloader, model, loss_fn, optimizer, output_option):
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X.unsqueeze_(0).permute(1, 0, 2))
        loss = loss_fn(pred, y) if output_option else loss_fn(pred, y.resize_(len(y), 1))
        
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 500 == 0:
            current = batch * batch_size + len(X)
            print(f"loss: {loss.item():>7f}  [{current:>5d}/{len(dataloader.dataset):>5d}]")

# -------------------------------
# Evaluation Loop
# -------------------------------
def test_loop(dataloader, model, loss_fn, output_option):
    model.eval()
    metric = MulticlassF1Score(average=None, num_classes=10)
    test_loss, correct = 0, 0
    category_correct, category_total = defaultdict(int), defaultdict(int)
    categories = np.arange(10)

    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X.unsqueeze_(0).permute(1, 0, 2))

            if output_option:
                test_loss += loss_fn(pred, y).item()
                preds = pred.argmax(1)
                true = y.argmax(1)
                correct += (preds == true).float().sum().item()
                metric.update(preds, true)
                for t, p in zip(true, preds):
                    category_total[t.item()] += 1
                    if t == p:
                        category_correct[t.item()] += 1
            else:
                test_loss += loss_fn(pred, y.resize_(len(y), 1)).item()

    accuracy = 100 * correct / len(dataloader.dataset)
    avg_loss = test_loss / len(dataloader)
    category_accuracy = {
        cat: (category_correct[cat] / category_total[cat]) * 100 if category_total[cat] else 0
        for cat in categories
    }

    print(f"\nTest Error: \n Accuracy: {accuracy:>0.1f}%, Avg loss: {avg_loss:>8f}")
    for category, acc in category_accuracy.items():
        print(f"Category {category}: {acc:.2f}% correct")
    print("\n")

    return avg_loss, metric.compute()

# -------------------------------
# Hyperparameters
# -------------------------------
learning_rate = 8e-3
batch_size = 64
epochs = 20

loss_fn = (
    nn.CrossEntropyLoss(weight=samples_weight) if output_option else nn.L1Loss()
)
optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)

# -------------------------------
# Data Preparation
# -------------------------------
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# -------------------------------
# Training Loop with Early Stopping
# -------------------------------
t = 0
test_loss = 0
prev_test_loss = np.inf

while test_loss - prev_test_loss < 0:
    t += 1
    if test_loss > 0:
        prev_test_loss = test_loss
    print(f"Epoch {t}\n-------------------------------")
    train_loop(train_loader, model, loss_fn, optimizer, output_option)
    test_loss, F1 = test_loop(test_loader, model, loss_fn, output_option)
    print(f"\nF1 Score: {F1}")

print("Done!")
