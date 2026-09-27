import torch
import torch.nn as nn
import torch.nn.functional as F


class Classifier(nn.Module):

    def __init__(
        self,
        embedding_dim=512,
        hidden_dim=128,
        num_classes=6
    ):
        super().__init__()

        self.fc = nn.Linear(
            embedding_dim,
            hidden_dim
        )

        self.classifier = nn.Linear(
            hidden_dim,
            num_classes
        )

    def forward(self, x):

        x = F.relu(
            self.fc(x)
        )

        return self.classifier(x)