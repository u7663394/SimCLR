from torchvision.transforms import transforms
from data_aug.gaussian_blur import GaussianBlur
from torchvision import transforms, datasets
from data_aug.view_generator import ContrastiveLearningViewGenerator
from exceptions.exceptions import InvalidDatasetSelection


class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        """
        初始化数据集类
        :param root_folder: 数据集存放的根目录
        """
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """
        构建 SimCLR 论文中的数据增强流程

        :param size: 图像裁剪后的尺寸 (例如 CIFAR10 是 32, STL10 是 96)
        :param s: 强度系数 (控制颜色扰动强度)
        :return: 一个组合好的数据增强 pipeline
        """
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s) # # 颜色扰动（亮度、对比度、饱和度、色调）
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size), # # 随机裁剪并缩放到指定尺寸
                                              transforms.RandomHorizontalFlip(), # 随机水平翻转
                                              transforms.RandomApply([color_jitter], p=0.8), # 以80%概率应用颜色扰动
                                              transforms.RandomGrayscale(p=0.2), # 以20%概率转为灰度图
                                              GaussianBlur(kernel_size=int(0.1 * size)), # 高斯模糊
                                              transforms.ToTensor()])
        return data_transforms

    def get_dataset(self, name, n_views):
        """
        根据数据集名称获取对应的数据集

        :param name: 数据集名称 ('cifar10' 或 'stl10')
        :param n_views: 每张图片生成多少个增强视图 (SimCLR 中一般为 2)
        :return: 数据集对象
        """
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, 
                                                              train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(32),
                                                                  n_views),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(self.root_folder, 
                                                          split='unlabeled', # 使用无标签数据 (自监督学习)
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(96),
                                                              n_views),
                                                          download=True)}

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()
