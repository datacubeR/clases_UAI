import torchmetrics
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(
    train_data,
    val_data,
    model,
    training_params,
    num_classes=10,
    criterion=nn.CrossEntropyLoss(),
):
    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=training_params["learning_rate"],
        weight_decay=training_params["weight_decay"],
    )

    train_dataloader = DataLoader(
        train_data,
        batch_size=training_params["batch_size"],
        shuffle=True,
        num_workers=12,
        pin_memory=True,
    )
    val_dataloader = DataLoader(
        val_data,
        batch_size=training_params["batch_size"],
        shuffle=False,
        num_workers=12,
        pin_memory=True,
    )

    train_metric = torchmetrics.F1Score(
        task="multiclass", num_classes=num_classes
    ).to(device)
    val_metric = torchmetrics.F1Score(
        task="multiclass", num_classes=num_classes
    ).to(device)

    train_loss = []
    val_loss = []
    for e in range(training_params["num_epochs"]):
        start_time = time.time()
        train_batch_loss = []
        val_batch_loss = []
        model.train()
        for batch in train_dataloader:
            X, y = batch["X"].to(device), batch["y"].to(device)

            optimizer.zero_grad()
            y_hat = model(X)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()
            tr_acc = train_metric(y_hat, y)
            train_batch_loss.append(loss.item())

        tr_acc = train_metric.compute()
        train_epoch_loss = np.mean(train_batch_loss)

        model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                X, y = batch["X"].to(device), batch["y"].to(device)
                y_hat = model(X)
                loss = criterion(y_hat, y)
                val_acc = val_metric(y_hat, y)
                val_batch_loss.append(loss.item())

        val_acc = val_metric.compute()
        val_epoch_loss = np.mean(val_batch_loss)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print(
            f"Epoch: {e+1}: Time: {elapsed_time:.2f} - Train Loss: {train_epoch_loss:.4f} - Validation Loss: {val_epoch_loss:.4f} - Train F1-Score: {tr_acc:.4f} - Validation F1-Score: {val_acc:.4f}"
        )
        train_loss.append(train_epoch_loss)
        val_loss.append(val_epoch_loss)

    return model, train_loss, val_loss
