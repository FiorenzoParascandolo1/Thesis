import torch
from torch import nn


class Baseline(nn.Module):
    """
    Baseline model.
    """

    def __init__(self,
                 input_size,
                 layers,
                 horizon):
        """
        :param input_size: past time window dimension
        :param layers: number of layers
        :param horizon: cardinality of space actions
        """
        super(Baseline, self).__init__()
        self.fc = nn.Sequential(*[nn.Sequential(nn.Linear(input_size, input_size), nn.ReLU()) for _ in range(layers)])
        self.classifier = nn.Linear(input_size, horizon)

    def forward(self,
                x: torch.Tensor) -> torch.Tensor:
        """
        :param x: input
        :return: predicted prices
        """
        return self.classifier(self.fc(x))
