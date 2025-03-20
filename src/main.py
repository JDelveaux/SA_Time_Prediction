# -*- coding: utf-8 -*-
"""
Loads the Datasets
Converts to PyTorch Format
Calls the NN

@author: jaked
"""

import torch
import numpy as np
from collections import defaultdict
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Lambda
from torcheval.metrics import MulticlassF1Score

#SET DEVICE
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

#SELECT DATASET
#for dsn in ["S2", "S4", "S5", "S6", "S7", "all"]:

dataset_name = "all"
conv = True
output_option = 1
weighted_output = False

#LOAD DATASETS
PATH = "../data/"
data = np.load(PATH + dataset_name + "_data_" + str(output_option) + ".npy", allow_pickle=True)

#CREATE PYTORCH DATASET
class SAMHSADataset(Dataset):
    def __init__(self, np_array, target_transform = None):
        self.data = np_array
        self.target_transform = target_transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        inp = torch.from_numpy(self.data[idx,1:].astype(float))
        #subtract 1 s.t. range 1-37 --> range 0-36
        if self.target_transform:
            label = self.data[idx,0] - 1
            label = self.target_transform(label)
        else:
            label = torch.tensor(self.data[idx,0] - 1)
        return inp.float(), label.float()

#Weight Sampling
class_sample_count = np.array(
    [len(np.where(data[:,0] == t)[0]) for t in np.unique(data[:,0])])
weight = 1. / class_sample_count
if weighted_output:
    samples_weight = torch.from_numpy(weight / weight.min())
else:
    samples_weight = None

if output_option:
    dataset = SAMHSADataset(data, target_transform = Lambda(lambda y: torch.zeros(10, dtype=torch.float).scatter_(0, torch.tensor(y), value=1)))
else:
    dataset = SAMHSADataset(data)

#CREATE NN
if dataset_name != "all":
    if conv:
        class NeuralNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv1d(1, 25, 5, padding = 3)
                self.pool = nn.MaxPool1d(5)
                self.conv2 = nn.Conv1d(25, 107, 5, padding = 3)
                self.fc1 = nn.Linear(1391, 1024)
                self.fc2 = nn.Linear(1024, 512)
                self.fc3 = nn.Linear(512, 10)

            def forward(self, x):
                x = self.pool(F.tanh(self.conv1(x)))
                x = self.pool(F.tanh(self.conv2(x)))
                x = torch.flatten(x, 1) # flatten all dimensions except batch
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x
    else:
        
        class NeuralNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.flatten = nn.Flatten()
                self.linear_stack = nn.Sequential(
                    nn.Linear(321, 512),
                    nn.ReLU(),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, 10)
                    )
            def forward(self, x):
                x = self.flatten(x)
                logits = self.linear_stack(x)
                return logits
            
elif dataset_name == "all":
    if conv:
        class NeuralNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv1d(1, 25, 5, padding = 3)
                self.pool = nn.MaxPool1d(5)
                self.conv2 = nn.Conv1d(25, 107, 5, padding = 3)
                self.fc1 = nn.Linear(1391, 1024)
                self.fc2 = nn.Linear(1024, 512)
                self.fc3 = nn.Linear(512, 10)

            def forward(self, x):
                x = self.pool(F.tanh(self.conv1(x)))
                x = self.pool(F.tanh(self.conv2(x)))
                x = torch.flatten(x, 1) # flatten all dimensions except batch
                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)
                return x
    else:
        class NeuralNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.flatten = nn.Flatten()
                self.linear_stack = nn.Sequential(
                    nn.Linear(329, 512),
                    nn.ReLU(),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, 10)
                    )
            def forward(self, x):
                x = self.flatten(x)
                logits = self.linear_stack(x)
                return logits
            
model = NeuralNetwork().to(device)

#TRAIN AND TEST LOOP
def train_loop(dataloader, model, loss_fn, optimizer, output_option):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X.unsqueeze_(0).permute(1, 0, 2))
        if output_option:
            loss = loss_fn(pred, y)
        else:
            loss = loss_fn(pred, y.resize_(len(y),1))

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 500 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn, output_option):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    metric = MulticlassF1Score(average=None, num_classes=10)
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    categories = np.arange(10)
    category_correct, category_total = defaultdict(int), defaultdict(int)
    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X.unsqueeze_(0).permute(1, 0, 2))
            if output_option:
                test_loss += loss_fn(pred, y).item()
                correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
                metric.update(pred.argmax(1), y.argmax(1))
                for true, pred in zip(y.argmax(1), pred.argmax(1)):
                    category_total[true.item()] += 1
                    if true == pred:
                        category_correct[true.item()] += 1
            else:
                test_loss += loss_fn(pred, y.resize_(len(y),1)).item()

    test_loss /= num_batches
    correct /= size
    category_accuracy = {cat: (category_correct[cat] / category_total[cat]) * 100 if category_total[cat] > 0 else 0 for cat in categories}
    print(f"\nTest Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f}")
    for category, accuracy in category_accuracy.items():
        print(f"Category {category}: {accuracy:.2f}% correct")
    print("\n")
    return test_loss, metric.compute()
    
#SET HYPERPARAMETERS, LOSS FN, OPTIMIZER
learning_rate = 8e-3
batch_size = 64
epochs = 20

if output_option:
    loss_fn = nn.CrossEntropyLoss(weight = samples_weight)
else:
    loss_fn = nn.L1Loss()

#optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)

#DEFINE TEST AND TRAINING SETS
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size

train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

#CREATE DATALOADERS
train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False)

#RUN NETWORK
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


    #del data