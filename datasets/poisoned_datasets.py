import random

import torch
from PIL import Image
from torchvision.datasets import MNIST, CIFAR10

from attack import TriggerHandler


class MINISTPoison(MNIST):
    """
    MINIST 数据集的毒化版本

    :param args: 命令行参数
    :param root: 数据集的根目录
    :param train: 是否为训练集，默认为 True
    :param transform: 数据集的变换
    :param target_transform: 数据集标签的变换
    :param download: 是否下载数据集，默认为 False
    """

    def __init__(self, args, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.width, self.height = self.__shape_info__()
        self.channels = 1

        self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.target_label, self.width,
                                              self.height, mode='L')

        print(f'pr={args.poisoning_rate} train?:{self.train}')
        self.poisoning_rate = args.poisoning_rate if train else 1.0  # 毒化率
        indices = range(len(self.targets))
        self.poison_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poison_indices)} over {len(indices)} samples (poisoning rate {self.poisoning_rate})")

    def __shape_info__(self):
        return self.data.shape[1:]

    def __getitem__(self, index):
        image, target = self.data[index], self.targets[index]
        image = Image.fromarray(image.numpy(), mode='L')

        if index in self.poison_indices:
            target = self.trigger_handler.target_label
            image = self.trigger_handler.add_trigger(image)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        target = torch.tensor(target)

        return image, target


class CIFAR10Poison(CIFAR10):
    """
    CIFAR10 数据集的毒化版本

   :param args: 命令行参数
   :param root: 数据集的根目录
   :param train: 是否为训练集，默认为 True
   :param transform: 数据集的变换
   :param target_transform: 数据集标签的变换
   :param download: 是否下载数据集，默认为 False
   """

    def __init__(self, args, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.width, self.height, self.channels = self.__shape_info__()

        self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.target_label, self.width,
                                              self.height, mode='L')

        print(f'pr={args.poisoning_rate} train?:{self.train}')
        self.poisoning_rate = args.poisoning_rate if train else 1.0  # 毒化率
        indices = range(len(self.targets))
        self.poison_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poison_indices)} over {len(indices)} samples (poisoning rate {self.poisoning_rate})")

    def __shape_info__(self):
        return self.data.shape[1:]

    def __getitem__(self, index):
        image, target = self.data[index], self.targets[index]
        image = Image.fromarray(image, mode='RGB')

        if index in self.poison_indices:
            target = self.trigger_handler.target_label
            image = self.trigger_handler.add_trigger(image)

        if self.transform is not None:
            image = self.transform(image)

        if self.target_transform is not None:
            target = self.target_transform(target)

        target = torch.tensor(target)

        return image, target
