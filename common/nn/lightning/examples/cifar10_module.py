import pytorch_lightning as pl
import torch
import torchvision.transforms as transforms
from common.constants import DATA_LONG_TERM_DIR
from common.experiments.config2args import make_config_loggable
from common.nn.models.image.cifar10_models import *
from torch import nn
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10


def get_classifier(classifier, pretrained):
    if classifier == "vgg11_bn":
        return vgg11_bn(pretrained=pretrained)
    elif classifier == "vgg13_bn":
        return vgg13_bn(pretrained=pretrained)
    elif classifier == "vgg16_bn":
        return vgg16_bn(pretrained=pretrained)
    elif classifier == "vgg19_bn":
        return vgg19_bn(pretrained=pretrained)
    elif classifier == "resnet18":
        return resnet18(pretrained=pretrained)
    elif classifier == "resnet34":
        return resnet34(pretrained=pretrained)
    elif classifier == "resnet50":
        return resnet50(pretrained=pretrained)
    elif classifier == "densenet121":
        return densenet121(pretrained=pretrained)
    elif classifier == "densenet161":
        return densenet161(pretrained=pretrained)
    elif classifier == "densenet169":
        return densenet169(pretrained=pretrained)
    elif classifier == "mobilenet_v2":
        return mobilenet_v2(pretrained=pretrained)
    elif classifier == "googlenet":
        return googlenet(pretrained=pretrained)
    elif classifier == "inception_v3":
        return inception_v3(pretrained=pretrained)
    else:
        raise NameError("Please enter a valid classifier")


class CIFAR10_Module(pl.LightningModule):
    def __init__(self, base_model: nn.Module, head_model_lambda, hparams, **kwargs):
        super().__init__()
        self.hparams = make_config_loggable(hparams)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.mean = [0.4914, 0.4822, 0.4465]
        self.std = [0.2023, 0.1994, 0.2010]
        self.model = get_classifier("resnet18", False)
        self.val_size = len(self.val_dataloader().dataset)

    def forward(self, batch):
        images, labels = batch
        predictions = self.model(images)
        loss = self.criterion(predictions, labels)
        accuracy = torch.sum(torch.max(predictions, 1)[1] == labels.data).float() / batch[0].size(0)
        return loss, accuracy

    def training_step(self, batch, batch_nb):
        loss, accuracy = self.forward(batch)
        logs = {"loss/train": loss, "accuracy/train": accuracy}
        return {"loss": loss, "log": logs}

    def validation_step(self, batch, batch_nb):
        avg_loss, accuracy = self.forward(batch)
        loss = avg_loss * batch[0].size(0)
        corrects = accuracy * batch[0].size(0)
        logs = {"loss/val": loss, "corrects": corrects}
        return logs

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x["loss/val"] for x in outputs]).sum() / self.val_size
        accuracy = torch.stack([x["corrects"] for x in outputs]).sum() / self.val_size
        logs = {"loss/val": loss, "accuracy/val": accuracy}
        return {"val_loss": loss, "progress_bar": {"accuracy/val": accuracy}, "log": logs}

    def test_step(self, batch, batch_nb):
        return self.validation_step(batch, batch_nb)

    def test_epoch_end(self, outputs):
        accuracy = self.validation_epoch_end(outputs)["log"]["accuracy/val"]
        accuracy = round((100 * accuracy).item(), 2)
        return {"progress_bar": {"Accuracy": accuracy}}

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-2, weight_decay=1e-2, momentum=0.9, nesterov=True)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        transform_train = transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )
        dataset = CIFAR10(root=DATA_LONG_TERM_DIR, train=True, transform=transform_train)
        dataloader = DataLoader(dataset, batch_size=128, num_workers=4, shuffle=True, drop_last=True, pin_memory=True)
        return dataloader

    def val_dataloader(self):
        transform_val = transforms.Compose([transforms.ToTensor(), transforms.Normalize(self.mean, self.std)])
        dataset = CIFAR10(root=DATA_LONG_TERM_DIR, train=False, transform=transform_val)
        dataloader = DataLoader(dataset, batch_size=1000, num_workers=4, shuffle=False, pin_memory=True)
        return dataloader

    def test_dataloader(self):
        return self.val_dataloader()
