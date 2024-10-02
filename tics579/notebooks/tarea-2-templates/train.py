import torch
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model(
    train_data,
    val_data,
    model,
    training_params,
    num_classes=10,
    criterion=nn.CrossEntropyLoss(),
):
    pass
