import torch.nn as nn

from models.encoder import Encoder
from models.projection_head import ProjectionHead
from models.classifier import Classifier
from models.wasserstein import WassersteinSupportPoints


class WASSModel(nn.Module):

    def __init__(
        self,
        embedding_dim=512,
        projection_head=None,
        num_classes=6,
        num_supports=1 #2,3,4,5,6……
    ):
        super().__init__()

        self.encoder = Encoder(
            embedding_dim=embedding_dim
        )

        self.projection_head = ProjectionHead(
            in_channels=embedding_dim,
            **(projection_head or {})
        )

        self.classifier = Classifier(
            embedding_dim=embedding_dim,
            hidden_dim=128,
            num_classes=num_classes
        )

        self.support_points = WassersteinSupportPoints(
            num_classes=num_classes,
            embedding_dim=embedding_dim,
            num_supports=num_supports
        )

    def forward(self, x):

        embedding = self.encoder(x)

        projection = self.projection_head(
            embedding
        )

        logits = self.classifier(
            embedding
        )

        supports = self.support_points()

        return (
            embedding,
            projection,
            logits,
            supports
        )
