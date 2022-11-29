import torchvision
from torch.utils.data import DataLoader
from model import *
import torch.nn.functional as F


class mnist():
    def __init__(self):
        self.train_data = torchvision.datasets.MNIST(
            root='./mnist',
            train=True,
            transform=torchvision.transforms.ToTensor(),
            # 把灰階從0~255壓縮到0~1
            download=True
        )
        self.test_data = torchvision.datasets.MNIST(
            root='./mnist',
            train=False,
            transform=torchvision.transforms.ToTensor(),
            # 把灰階從0~255壓縮到0~1
            download=True
        )
        self.batch_size = 64
        self.model = ConvNet()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.data_train, self.data_test = self.batch_data()
        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.epoch = 5

    def batch_data(self):
        train_dataloader = DataLoader(self.train_data, batch_size=self.batch_size, shuffle=False)
        test_dataloader = DataLoader(self.test_data, batch_size=len(self.test_data), shuffle=False)
        return train_dataloader,test_dataloader

    def train(self):
        for batch,(data, label) in enumerate(self.data_train):
            out = self.model(data).to(self.device)
            loss = self.loss(out, label.to(self.device))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if batch % 100 == 0:
                loss, current = loss.item(), batch * len(data)
                print(f"loss: {loss:>7f}  [{current:>5d}/{len(self.train_data):>5d}]")

    def test(self):
        test_loss, correct = 0, 0
        # self.model.eval()
        with torch.no_grad():
            for X, y in self.data_test:
                pred = self.model(X).to(self.device)
                test_loss += self.loss(pred, y.to(self.device)).item()

                correct += (pred.argmax(1) == y.to(self.device)).type(torch.float).sum().item()
        test_loss /= len(self.test_data)
        correct /= len(self.test_data)
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

    def step(self):
        for epoch in range(self.epoch):
            print('epoch',epoch)
            self.train()
        self.test()


if __name__ == '__main__':
    mnist = mnist()
    mnist.step()
