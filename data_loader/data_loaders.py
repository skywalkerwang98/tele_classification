from torchvision import datasets, transforms
from base import BaseDataLoader


class MnistDataLoader(BaseDataLoader):
    """
    MNIST data loading demo using BaseDataLoader
    """
    def __init__(self, data_dir, batch_size, shuffle=True, validation_split=0.0, num_workers=1, training=True):
        trsfm = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.data_dir = data_dir
        self.dataset = datasets.MNIST(self.data_dir, train=training, download=True, transform=trsfm)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers)

class ConstellationGraphDataLoader(BaseDataLoader):
    """
    Constellation Graph data loader
    """
    def __init__(self, batch_size=32, shuffle=True, validation_split=0.2, num_workers=1, training=True):
        transformers = transforms.Compose([
            transforms.ToTensor(),
            # TODO: 进一步统计星座数据集的均值和方差，然后用这个均值和方差来做归一化
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        from data_loader.my_datasets import ConstellationGraphDatasets
        self.dataset = ConstellationGraphDatasets(train=training, transform=transformers)
        super().__init__(self.dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=self.dataset.collate_fn)
