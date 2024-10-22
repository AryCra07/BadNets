import torch
import torch.nn as nn
import torch.nn.functional as F


class BadNet(nn.Module):
    """
    BadNet 是用于图像分类任务的神经网络模型，由三个卷积层和两个全连接层组成。
    :param input_channels: 图像的通道数
    :param output_classes:网络的输出类别数
    """
    def __init__(self, input_channels, output_classes):
        super(BadNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=input_channels, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(p=.25)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        fc1_input_features = 64 * 4 * 4 if input_channels == 3 else 64 * 3 * 3
        self.fc1 = nn.Linear(in_features=fc1_input_features, out_features=512)
        self.fc2 = nn.Linear(512, output_classes)
        self.dropout_fc = nn.Dropout(p=.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))

        x = self.dropout(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout_fc(x)

        x = self.fc2(x)

        x = F.log_softmax(x, dim=1)

        return x


# 测试模型的定义
if __name__ == "__main__":
    # 创建一个模型实例
    model = BadNet(input_channels=3, output_classes=10)
    # 打印模型结构
    print(model)

    # 测试输入（假设输入是一个批量大小为8的32x32 RGB图像）
    test_input = torch.randn(8, 3, 32, 32)
    test_output = model(test_input)
    print(f"输出形状: {test_output.shape}")
