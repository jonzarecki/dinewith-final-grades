from collections import OrderedDict

import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_dims, n_hiddens):
        super().__init__()
        assert isinstance(input_dims, int), "Please provide int for input_dims"
        self.input_dims = input_dims
        current_dims = input_dims
        layers = OrderedDict()

        if isinstance(n_hiddens, int):
            n_hiddens = [n_hiddens]
        else:
            n_hiddens = list(n_hiddens)
        for i, n_hidden in enumerate(n_hiddens):
            layers[f"fc{i + 1}"] = nn.Linear(current_dims, n_hidden)
            layers[f"relu{i + 1}"] = nn.ReLU()
            layers[f"drop{i + 1}"] = nn.Dropout(0.2)
            current_dims = n_hidden

        self.model = nn.Sequential(layers)

    def forward(self, input):
        input = input.view(input.size(0), -1)
        assert input.size(1) == self.input_dims
        return self.model.forward(input)


def mnist_mlp(input_dims=784, n_hiddens=[128, 128]):
    model = MLP(input_dims, n_hiddens)
    return model
