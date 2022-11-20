import torch
from torch.utils.data import Dataset
from PIL import Image
import csv

class ConstellationGraphDatasets(Dataset):
    """自定义数据集"""
    def __init__(self, train=True, transform=None):
        if train:
            self.dataset_info_file = 'data/constellationGraph_info_train.csv'
            self.collate_fn = self.collate_fn_train
        else:
            self.dataset_info_file = 'data/constellationGraph_info_test.csv'
            self.collate_fn = self.collate_fn_test
        self.transform = transform
        self.image_paths, self.image_labels, self.image_noisy_values = self.read_image_datasets_info_file()
        
    def read_image_datasets_info_file(self):
        # TODO：修改为参数列表形式到utils，方便复用
        image_paths = []
        image_labels = []
        image_noisy_values = []
        with open(self.dataset_info_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                image_paths.append(row[0])
                image_labels.append(row[1])
                image_noisy_values.append(row[2])
        return image_paths, image_labels, image_noisy_values

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, item):
        img = Image.open(self.image_paths[item])
        img = img.convert('L')
        #img.show()
        # RGB为彩色图片，L为灰度图片
        # if img.mode != 'RGB':
        #     raise ValueError("image: {} isn't RGB mode.".format(self.image_paths[item]))
        label = self.image_labels[item]
        noisy_value = self.image_noisy_values[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, int(label), float(noisy_value)

    def __len__(self):
        return len(self.image_paths)

    @staticmethod
    def collate_fn_train(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        # 拼接为tensor
        # TODO:因为数据集额外返回了noisy value，这里暂时留空保证可以运行
        images, labels, _ = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels

    @staticmethod
    def collate_fn_test(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        # 拼接为tensor
        # TODO:因为数据集额外返回了noisy value，这里暂时留空保证可以运行
        images, labels, noisy_value = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels, noisy_value

if __name__ == "__main__":
    # 测试代码
    train_dataset = ConstellationGraphDatasets(train=True, transform=None)
    test_dataset = ConstellationGraphDatasets(train=False, transform=None)
    print(f'训练集大小：{len(train_dataset)}')
    print(f'测试集大小：{len(test_dataset)}')
    print(f'训练集第一张图片：{train_dataset[0]}')
    print(f'测试集第一张图片：{test_dataset[0]}')
    print('训练集测试开始')
    for index in range(len(train_dataset)):
        img, label, noisy_value = train_dataset[index]
    print('训练集测试通过')
    print('测试集测试开始')
    for index in range(len(test_dataset)):
        img, label, noisy_value = test_dataset[index]
    print('测试集测试通过')
