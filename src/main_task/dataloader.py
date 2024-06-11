from torch.utils.data import Dataset

class ArticleDataset(Dataset):
    def __init__(self, titles):
        self.titles = titles

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        return self.titles[idx]
