from torchmeta.datasets.helpers import omniglot
from torchmeta.utils.data import BatchMetaDataLoader
from model import ConvNet


class maml():
    def __init__(self):
        self.dataset = omniglot("data", ways=5, shots=5, test_shots=15, meta_train=True, download=True)
        self.dataloader = BatchMetaDataLoader(self.dataset, batch_size=16, num_workers=4)
        self.model = ConvNet()


if __name__ == '__main__':
    maml = maml()
    for batch in maml.dataloader:
        train_data,train_target=batch['train']
        # print(train_data.shape)
        out=maml.model(train_data)
        print(out)
        break
