from torch.utils.data.dataset import Dataset


class MyDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.size(dim=0)

    def __getitem__(self, index):
        return self.x[index], self.y[index]