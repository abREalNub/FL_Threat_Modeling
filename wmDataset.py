from torch.utils.data import Dataset, Subset

class wmDataset(Dataset):
    def __init__(self, dataset= None, transform=None):
        self.dataset = dataset
        self.transform=transform
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        image, label = self.dataset[item]
        self.transform=None
        if self.transform != None:
            image = self.transform(image)
        return image, label