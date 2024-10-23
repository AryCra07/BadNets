import random

from PIL import Image
from torchvision.datasets import MNIST


class TriggerHandler:
    """
    触发器处理类，用于在图像中添加触发器

    :param trigger_path: 触发器图片的路径
    :param trigger_size: 触发器的大小
    :param target_label: 触发器的标签
    :param img_width: 图像的宽度
    :param img_height: 图像的高度
    :param mode: 图像的模式，默认为 RGB
    """

    def __init__(self, trigger_path, trigger_size, target_label, img_width, img_height, mode='RGB'):
        self.mode = mode
        self.trigger_img = Image.open(trigger_path).convert(self.mode)
        self.trigger_size = trigger_size
        self.trigger_img = self.trigger_img.resize((trigger_size, trigger_size))
        self.target_label = target_label
        self.img_width = img_width
        self.img_height = img_height

    def add_trigger(self, image):
        # 在图像的右下角添加触发器
        image.paste(self.trigger_img, (self.img_width - self.trigger_size, self.img_height - self.trigger_size))

        return image
