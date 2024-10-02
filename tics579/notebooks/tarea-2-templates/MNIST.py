from sklearn.datasets import fetch_openml

from torch.utils.data import Dataset

SEED = 10

df, target = fetch_openml("mnist_784", return_X_y=True)
print(f"Shape X: {df.shape}")
print(f"Shape y: {target.shape}")


class MNIST(Dataset):
    def __init__(self, X, y):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
