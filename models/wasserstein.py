import torch
import torch.nn as nn
import torch.nn.functional as F


class WassersteinSupportPoints(nn.Module):
    def __init__(
        self,
        num_classes,
        embedding_dim=512,
        num_supports=8
    ):
        super().__init__()

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.num_supports = num_supports

        self.support_points = nn.Parameter(
            torch.randn(
                num_classes,
                num_supports,
                embedding_dim
            )
        )

        nn.init.normal_(
            self.support_points,
            mean=0.0,
            std=0.02
        )

    def forward(self):
        return F.normalize(
            self.support_points,
            p=2,
            dim=-1
        )

    def get_class_supports(self, class_id):

        return self.forward()[class_id]

    def extra_repr(self):

        return (
            f"num_classes={self.num_classes}, "
            f"num_supports={self.num_supports}, "
            f"embedding_dim={self.embedding_dim}"
        )