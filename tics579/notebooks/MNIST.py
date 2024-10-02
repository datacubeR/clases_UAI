import torch
from torch.utils.data import Dataset
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

SEED = 10

df, target = fetch_openml("mnist_784", return_X_y=True)
print(f"Shape X: {df.shape}")
print(f"Shape y: {target.shape}")

X_train, X_val, y_train, y_val = train_test_split(
    df, target, test_size=10000, random_state=SEED
)
X_train.shape, X_val.shape, y_train.shape, y_val.shape


class MNIST(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.values).float()
        self.y = torch.from_numpy(y.astype("float32").values).long()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return dict(X=self.X[idx], y=self.y[idx])


train_data = MNIST(X_train, y_train)
val_data = MNIST(X_val, y_val)

print(f"Length Pytorch X: {len(train_data)}")
print(f"Length Pytorch X: {len(val_data)}")
