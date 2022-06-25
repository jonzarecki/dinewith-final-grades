import copy

import torch.nn as nn
from common.datasets.cifar import cifar_transforms
from common.datasets.nist import nist_transforms
from common.datasets.stanford_cars import stanford_cars
from common.nn.losses import LabelSmoothingCrossEntropy
from common.nn.models.general.mlp import mnist_mlp
from common.nn.models.image.resnet import ResNet50, ResNet101
from common.nn.models.image.resnet_pt import resnet18
from common.nn.models.image.vgg import VGG
from common.nn.models.image.wideresnet import wresnet28_10, wresnet40_2
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, CyclicLR, MultiStepLR, OneCycleLR
from torch.utils.data import Subset
from torchvision import datasets
from torchvision.models import densenet169


def add_to_config(args):
    args = add_dataset_to_config(args)
    args = add_models_to_config(args)
    args = add_optimizers_to_config(args)

    return args


def add_dataset_to_config(args):
    if args.dataset_name == "mnist":
        args.dataset = datasets.MNIST
        args.lbl_num = 10
        args.in_channels = 1
        args.transforms = nist_transforms
    elif args.dataset_name == "3class-mnist":

        def filter_to_3_classes(*args, **kwargs):
            dataset = datasets.MNIST(*args, **kwargs)
            targets = dataset.targets
            # Get indices to keep
            idx_to_keep = [n for n, t in enumerate(targets) if t.item() in (0, 1, 2)]
            # Only keep your desired classes
            filtered_dataset = Subset(dataset, idx_to_keep)
            filtered_dataset.classes = (0, 1, 2)
            return filtered_dataset

        args.dataset = filter_to_3_classes
        args.lbl_num = 3
        args.in_channels = 1
        args.transforms = nist_transforms
    elif args.dataset_name == "fashion-mnist":
        args.dataset = datasets.FashionMNIST
        args.lbl_num = 10
        args.in_channels = 1
        args.transforms = nist_transforms
    elif args.dataset_name == "kmnist":
        args.dataset = datasets.KMNIST
        args.lbl_num = 10
        args.in_channels = 1
        args.transforms = nist_transforms
    elif "cifar" in args.dataset_name:
        if args.dataset_name == "cifar10":
            args.dataset = datasets.CIFAR10
            args.lbl_num = 10
        elif args.dataset_name == "cifar100":
            args.dataset = datasets.CIFAR100
            args.lbl_num = 100

        args.in_channels = 3
        args.transforms = copy.deepcopy(cifar_transforms)
    elif args.dataset_name == "stanford-cars":
        args.dataset = stanford_cars.StanfordCarsDataset
        args.lbl_num = stanford_cars.class_num
        args.in_channels = 3
        args.transforms = {"train": stanford_cars.larger_train_transform, "val": stanford_cars.larger_test_transform}
    # elif args.dataset_name == 'mini-imagenet':
    # elif args.dataset_name == 'omniglot':
    # elif args.dataset_name == 'trec':  # text dataset

    return args


def add_models_to_config(args):
    if args.model_name == "mlp":

        args.model_lambda = lambda: mnist_mlp()
        args.head_lambda = lambda l_num: nn.Sequential(nn.Linear(128, l_num))
    elif args.model_name == "simple-cnn":
        from common.nn.models import Flatten, conv_block

        args.model_lambda = lambda: nn.Sequential(
            conv_block(args.in_channels, 32),
            conv_block(32, 24),
            conv_block(24, 16),
            Flatten(),
        )
        args.head_lambda = lambda l_num: nn.Sequential(nn.Linear(144, l_num))
        # args.model_lambda = \
        #     lambda: CNN()
        # args.head_lambda = lambda l_num: nn.Sequential(nn.Linear(256, l_num))
    elif args.model_name == "vgg8b":
        args.model_lambda = lambda: VGG("VGG8B_256", in_channels=args.in_channels)
        args.head_lambda = lambda l_num: nn.Sequential(nn.Linear(256, l_num))
    elif args.model_name == "densenet-bc-169":
        args.model_lambda = lambda: densenet169(num_classes=512)
        args.head_lambda = lambda l_num: nn.Sequential(nn.Linear(512, l_num))
    elif args.model_name == "resnet101":
        args.model_lambda = lambda: ResNet101()
        args.head_lambda = lambda l_num: nn.Sequential(nn.Linear(512, l_num))
    elif args.model_name == "resnet50":
        args.model_lambda = lambda: ResNet50()
        args.head_lambda = lambda l_num: nn.Sequential(nn.Linear(512, l_num))
    elif args.model_name == "wresnet28_10":
        args.model_lambda = lambda: wresnet28_10()
        args.head_lambda = lambda l_num: nn.Sequential(nn.Linear(64 * 10, l_num))
    elif args.model_name == "wresnet40_2":
        args.model_lambda = lambda: wresnet40_2()
        args.head_lambda = lambda l_num: nn.Sequential(nn.Linear(64 * 2, l_num))
    elif args.model_name in ("vgg11", "resnet18"):
        if args.model_name == "vgg11":
            args.model_lambda = lambda: VGG("VGG11", in_channels=args.in_channels)
            # args.model_lambda = lambda: VGG('VGG11', in_channels=(1 if args.dataset_name == 'MNIST' else 3))
        else:

            args.model_lambda = lambda: resnet18()  # ResNet18(args.in_channels)  # ConvolutionalNeuralNetwork(1, 64)
        args.head_lambda = lambda l_num: nn.Sequential(nn.Linear(512, l_num))
    # elif args.model_name == 'bert-gru':
    #     args.model_lambda = lambda: BertGRU()
    #     args.head_lambda = lambda l_num: nn.Sequential(nn.Linear(256, l_num))

    if args.loss == "ce":
        args.loss_lambda = lambda *args: nn.CrossEntropyLoss()
    elif args.loss == "ce-ls":
        args.loss_lambda = lambda *args: LabelSmoothingCrossEntropy()
    # elif args.loss == 'bge-logit':  # for text model
    #     args.loss_lambda = lambda *args: nn.BCEWithLogitsLoss()
    return args


def add_optimizers_to_config(args):
    if args.optimizer_name == "sgd":
        args.optimizer_lambda = lambda model: optim.SGD(
            model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-2, nesterov=True
        )
    elif args.optimizer_name == "adam":
        args.optimizer_lambda = lambda model: optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))
    elif args.optimizer_name == "adamw":
        args.optimizer_lambda = lambda model: optim.AdamW(model.parameters(), lr=args.lr)
    elif args.optimizer_name == "adagrad":
        args.optimizer_lambda = lambda model: optim.Adagrad(model.parameters(), lr=args.lr)
    elif args.optimizer_name == "rmsprop":
        args.optimizer_lambda = lambda model: optim.RMSprop(model.parameters(), lr=args.lr)
    elif args.optimizer_name == "adadelta":
        args.optimizer_lambda = lambda model: optim.Adadelta(model.parameters(), lr=args.lr)

    # lr scheduling
    if args.lr_sched_type == "constant":
        args.lr_scheduler_lambda = lambda opt, **kwargs: MultiStepLR(opt, milestones=[1], gamma=1.0)
    elif args.lr_sched_type == "multi-step":
        args.lr_scheduler_lambda = lambda opt, **kwargs: MultiStepLR(
            opt, milestones=args.lr_sched_milestones, gamma=args.lr_sched_gamma
        )
    elif args.lr_sched_type == "cyclic":
        args.lr_scheduler_lambda = lambda opt, **kwargs: CyclicLR(
            opt, step_size_up=int(args.epochs / 2), base_lr=1e-4, max_lr=5e-4
        )
    elif args.lr_sched_type == "one-cycle":
        args.lr_scheduler_lambda = lambda opt, steps_per_epoch, **kwargs: OneCycleLR(
            opt, max_lr=args.lr, steps_per_epoch=steps_per_epoch, epochs=args.epochs
        )
    elif args.lr_sched_type == "cosine":
        args.lr_scheduler_lambda = lambda opt, **kwargs: CosineAnnealingLR(opt, T_max=args.epochs, eta_min=0.0)

    return args
