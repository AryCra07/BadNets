from torchvision import transforms, datasets
from .poisoned_datasets import MINISTPoison


def init_dataset(name, path, is_download):
    if name == 'MNIST':
        train_data = MINISTPoison(root=path, train=True, download=is_download)
        test_data = MINISTPoison(root=path, train=False, download=is_download)
    return train_data, test_data


def attack_train_set(is_train, args):
    """
    用于构建中毒的训练集
    :param is_train: 是否为训练集
    :param args: 命令行参数
    :return: 训练集和输出类别数
    """
    transform, _ = build_transforms(args.dataset)
    train_set = MINISTPoison(args, args.data_path, train=is_train, download=True, transform=transform)
    output_classes = 10
    return train_set, output_classes


def test_set(is_train, args):
    """
    用于构建测试集
    :param is_train: 是否为训练集
    :param args: 命令行参数
    :return: 测试集和中毒测试集
    """
    transform, _ = build_transforms(args.dataset)
    set_clean = datasets.MNIST(args.data_path, train=is_train, download=True, transform=transform)
    set_poisoned = MINISTPoison(args, args.data_path, train=is_train, download=True, transform=transform)
    nb_classes = 10
    return set_clean, set_poisoned


def build_transforms(dataset):
    if dataset == "MNIST":
        mean, std = (0.1307,), (0.3081,)
    else:
        raise NotImplementedError()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    detransform = transforms.Compose([
        transforms.Normalize(mean=(-m / s for m, s in zip(mean, std)), std=(1 / s for s in std)),
    ])

    return transform, detransform
