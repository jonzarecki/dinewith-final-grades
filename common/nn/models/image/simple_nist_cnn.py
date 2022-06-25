import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, optim
from torchvision import datasets, transforms


class Flatten(nn.Module):
    """Converts N-dimensional Tensor of shape [batch_size, d1, d2, ..., dn] to 2-dimensional Tensor
    of shape [batch_size, d1*d2*...*dn].
    # Arguments
        input: Input tensor
    """

    def forward(self, input):
        return input.view(input.size(0), -1)


def conv_block(in_channels: int, out_channels: int) -> nn.Module:
    """Returns a Module that performs 3x3 convolution, ReLu activation, 2x2 max pooling.
    # Arguments
        in_channels:
        out_channels:
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


def build_model(class_num):
    return nn.Sequential(conv_block(1, 16), conv_block(16, 8), Flatten(), nn.Linear(128, class_num))


class CNN(nn.Module):
    """
    conv1 (channels = 10, kernel size= 5, stride = 1) -> Relu -> max pool (kernel size = 2x2) ->
    conv2 (channels = 50, kernel size= 5, stride = 1) -> Relu -> max pool (kernel size = 2x2) ->
    Linear (256) -> Relu
    Reshaping outputs from the last conv layer prior to feeding them into the linear layers.
    """

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(
            1, 10, 5, stride=1
        )  # Applies a 1D convolution over an input signal composed of several input planes
        self.maxp = nn.MaxPool2d(2, 2)  # MaxPool function
        self.conv2 = nn.Conv2d(
            10, 50, 5, stride=1
        )  # Applies a 1D convolution over an input signal composed of several input planes
        self.fc1 = nn.Linear(800, 256)  # the Linear value change to 256

    def forward(self, x):
        x = F.relu(self.conv1(x))  # convolution it

        x = self.maxp(x)  # pool it

        x = F.relu(self.conv2(x))  # convolution it

        x = self.maxp(x)  # pool it again

        x = x.view(x.shape[0], -1)  # make sure inputs are flattened
        x = F.relu(self.fc1(x))  # change the value to 256

        return x
