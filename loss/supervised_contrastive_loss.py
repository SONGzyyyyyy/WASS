import torch
import torch.nn.functional as F


def supervised_contrastive_loss(
        features_1,
        features_2,
        labels,
        temperature=0.07,
        eps=1e-8
):

    batch_size = features_1.size(0)
    device = features_1.device

    features = torch.cat(
        [features_1, features_2],
        dim=0
    )

    features = F.normalize(
        features,
        p=2,
        dim=1
    )

    similarity = torch.matmul(
        features,
        features.T
    ) / temperature

    # Numerical stability
    logits_max, _ = torch.max(
        similarity,
        dim=1,
        keepdim=True
    )

    logits = similarity - logits_max.detach()

    labels = torch.cat(
        [labels, labels],
        dim=0
    )

    # Positive mask
    positive_mask = torch.eq(
        labels.unsqueeze(1),
        labels.unsqueeze(0)
    ).float()

    # Remove self-comparison
    self_mask = torch.eye(
        2 * batch_size,
        device=device
    )

    positive_mask = (
        positive_mask
        * (1.0 - self_mask)
    )

    # Denominator
    exp_logits = (
        torch.exp(logits)
        * (1.0 - self_mask)
    )

    log_prob = logits - torch.log(
        exp_logits.sum(
            dim=1,
            keepdim=True
        ) + eps
    )

    num_positives = positive_mask.sum(
        dim=1
    ).clamp(min=eps)

    mean_log_prob_pos = (
        positive_mask * log_prob
    ).sum(dim=1) / num_positives

    return -mean_log_prob_pos.mean()