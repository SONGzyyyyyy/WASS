import torch.nn as nn

from models.mlp_head import MLPHead


class ProjectionHead(nn.Module):

    def __init__(
        self,
        in_channels=512,
        **kwargs
    ):
        super().__init__()

        self.projection = MLPHead(
            in_channels=in_channels,
            **kwargs
        )

    def forward(self, x):
        return self.projection(x)