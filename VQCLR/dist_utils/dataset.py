from torchvision import transforms, datasets
from cv_lib.classification.data.imagenet import ImageNet
from cv_lib.classification.data.caltech_101 import Caltech_101


def get_train_dataset():
    trainset = ImageNet('/mnt/datasets/ILSVRC2012', resize=224)
    return trainset


def get_test_dataset():
    testset = ImageNet('/mnt/datasets/ILSVRC2012', split='val', resize=224)
    return testset


def get_train_dataset_caltech101(data_path='/mnt/datasets/caltech-101'):
    trainset = Caltech_101(data_path, resize=224)
    return trainset


def get_test_dataset_caltech101(data_path='/mnt/datasets/caltech-101'):
    testset = Caltech_101(data_path, split='test', resize=224)
    return testset
