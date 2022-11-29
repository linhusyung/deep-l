import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from model import ConvNet


class Data_preprocessing(Dataset):
    def __init__(self):
        self.train_data = torchvision.datasets.Omniglot(
            root="./data", download=True, background=True, transform=torchvision.transforms.ToTensor()
        )
        self.text_data = torchvision.datasets.Omniglot(
            root="./data", download=True, background=False, transform=torchvision.transforms.ToTensor()
        )
        self.model = ConvNet()
        self.batch_size = 10

    def K_shot(self, data, K_shot):
        for d, l in data:
            pass

    def N_way_K_shot(self, data, N_way, K_shot):
        pass


if __name__ == '__main__':
    pre = Data_preprocessing()
    last = None
    for d, l in pre.text_data:
        torch.tensor(l)