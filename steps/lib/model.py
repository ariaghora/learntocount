from typing import List

import torch
from torch.nn import (
    BatchNorm2d,
    Flatten,
    LazyConv2d,
    LazyLinear,
    MaxPool2d,
    LeakyReLU,
    Sequential,
    Dropout,
)


class Model(torch.nn.Module):
    def __init__(self, n_views: int, n_classes: int) -> None:
        super().__init__()
        self.n_views = n_views
        self.n_classes = n_classes

        self.conv_blocks = [
            Sequential(
                LazyConv2d(32, 3, padding=1),
                BatchNorm2d(32),
                LeakyReLU(),
                MaxPool2d(2),
                Flatten(),
                LazyLinear(100),
                LeakyReLU(),
            )
            for _ in range(n_views)
        ]

        self.classifier = Sequential(
            LazyLinear(100),
            LeakyReLU(),
            Dropout(),
            LazyLinear(self.n_classes),
        )

    def forward(self, views: List[torch.Tensor]) -> torch.Tensor:
        embeddings = 0
        for i, view in enumerate(views):
            embeddings += self.conv_blocks[i](view)
        embeddings /= self.n_views
        output = self.classifier(embeddings)

        return output

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
