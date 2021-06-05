from torch.utils import data
from torch.utils.data import Dataset
import torchvision.datasets as datasets
from torchvision.datasets.utils import download_and_extract_archive

class DataSet(Dataset):
    def __init__(self, train=False, transform=None):
        self.dataset = datasets.MNIST(root='./data', train=train, download=True, transform=transform)

    def __getitem__(self, index):
        data, target = self.dataset[index]
        return data, target

    def __len__(self): 
        return len(self.dataset)
