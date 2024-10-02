import torch.nn as nn


class MLP(nn.Module):
    def __init__(
        self,
        hidden_dims=[10, 20, 10],
        out_dim=1,
        bn=None,
        activation="relu",
        dropout=None,
    ):
        super().__init__()

        activations = nn.ModuleDict(
            dict(relu=nn.ReLU(), sigmoid=nn.Sigmoid())
        )
        self.encoder = nn.Sequential(
            *[
                self.dense_block(
                    in_dim, out_dim, activations[activation], bn, dropout
                )
                for in_dim, out_dim in zip(hidden_dims, hidden_dims[1:])
            ]
        )
        self.output = nn.Linear(hidden_dims[-1], out_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = self.output(x)
        return x

    @staticmethod
    def dense_block(in_dim, out_dim, activation, bn=None, dropout=None):
        pre = nn.ModuleDict(
            dict(pre=nn.BatchNorm1d(out_dim), post=nn.Identity())
        )
        post = nn.ModuleDict(
            dict(pre=nn.Identity(), post=nn.BatchNorm1d(out_dim))
        )

        return nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.Identity() if bn is None else pre[bn],
            activation,
            nn.Identity() if bn is None else post[bn],
            (
                nn.Identity()
                if dropout is None
                else nn.Dropout(dropout, inplace=True)
            ),
        )
