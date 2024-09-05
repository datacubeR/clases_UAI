# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
np.random.seed(1)
# %% Datos
X_numpy = np.random.randn(1000, 10)
y_numpy = np.random.randint(2, size=1000)

X = torch.from_numpy(X_numpy).float()
y = torch.from_numpy(y_numpy).float()
X.shape, y.shape

# %% Red Neuronal MLP

class MLP(nn.Module):
    def __init__(self, n_features, hidden_dim_1, hidden_dim_2, output_dim):
        super(MLP, self).__init__()
        self.w1 = nn.Linear(n_features, hidden_dim_1)
        self.w2 = nn.Linear(hidden_dim_1, hidden_dim_2)
        self.w3 = nn.Linear(hidden_dim_2,output_dim)
        self.relu_1 = nn.ReLU(inplace=True)
        self.relu_2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.w1(x)
        x = self.relu_1(x)
        x = self.w2(x)
        x = self.relu_2(x)
        x = self.w3(x)
        return x


model = MLP(n_features=10, hidden_dim_1=32, hidden_dim_2=64, output_dim=1)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)

# %% Entrenamiento
EPOCHS = 1000
for e in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    preds = model(X)
    loss = criterion(preds, y.unsqueeze(1))
    loss.backward()
    optimizer.step()
    print(f"Epoch {e} Loss: {loss.item()}")

# %%
from torch.utils.data import Dataset, DataLoader
class MyDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = MyDataset(X, y)
train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

EPOCHS = 1000

train_loss = np.empty(EPOCHS)
for e in range(EPOCHS):
    train_batch_loss = []
    for X_batch, y_batch in train_loader:
        model.train()
        optimizer.zero_grad()
        preds = model(X_batch)
        loss = criterion(preds, y_batch.unsqueeze(1))
        loss.backward()
        optimizer.step()
        train_batch_loss.append(loss.item())

    train_loss[e] = np.mean(train_batch_loss)
    print(f"Epoch {e} Loss: {train_loss[e]}")

import matplotlib.pyplot as plt

plt.plot(range(EPOCHS), train_loss)
