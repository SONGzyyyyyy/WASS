import torch.nn.functional as F


def classification_loss(logits, labels):
    """
    Cross-entropy classification loss.
    """

    return F.cross_entropy(
        logits,
        labels
    )