import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        hidden_dims,
        out_dim,
        bn,
        activation,
        dropout,
    ):
        pass

    def forward(self, x):
        pass

    @staticmethod
    def dense_block(in_dim, out_dim, activation, bn, dropout):
        pass
