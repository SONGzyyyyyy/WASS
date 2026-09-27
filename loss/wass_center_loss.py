import torch
from geomloss import SamplesLoss

def build_sinkhorn_loss(
        blur=0.05,
        p=2
):
    return SamplesLoss(
        loss="sinkhorn",
        p=p,
        blur=blur
    )


def _aggregate_support_points(support_points):
    if support_points.dim() == 2:
        return support_points

    if support_points.dim() == 3:
        return support_points.mean(dim=1)

    raise ValueError(
        f"support_points must have shape [C, D] or [C, M, D], "
        f"but got {tuple(support_points.shape)}"
    )


def prototype_wass_loss(
        features,
        support_points,
        labels,
        domains
):

    num_classes = support_points.size(0)
    device = features.device

    unique_domains = torch.unique(
        domains,
        sorted=True
    )

    num_domains = unique_domains.numel()

    # [N, Dm]
    domain_mask = (
        domains.unsqueeze(1)
        == unique_domains.unsqueeze(0)
    )

    # [N, C]
    class_ids = torch.arange(
        num_classes,
        device=device,
        dtype=labels.dtype
    )

    class_mask = (
        labels.unsqueeze(1)
        == class_ids.unsqueeze(0)
    )

    # [N, Dm, C]
    valid_mask = (
        domain_mask.unsqueeze(2)
        & class_mask.unsqueeze(1)
    )

    # [Dm, C]
    counts = valid_mask.sum(dim=0).float()

    # [Dm, C, D]
    feature_sums = torch.einsum(
        "ndc,ni->dci",
        valid_mask.float(),
        features
    )

    # Convert multi-support representation to one prototype.
    class_prototypes = _aggregate_support_points(
        support_points
    )

    # [Dm, C, D]
    fallback_prototypes = (
        class_prototypes.detach()
        .unsqueeze(0)
        .expand(num_domains, -1, -1)
    )

    domain_prototypes = torch.where(
        counts.unsqueeze(-1) > 0,
        feature_sums
        / counts.unsqueeze(-1).clamp_min(1e-6),
        fallback_prototypes
    )

    # [Dm, C]
    prototype_distances = torch.norm(
        domain_prototypes
        - class_prototypes.unsqueeze(0),
        p=2,
        dim=-1
    )

    # Average over classes first, then domains.
    prototype_loss = (
        prototype_distances.mean(dim=1).mean()
    )

    return prototype_loss


def sinkhorn_wass_loss(
        features,
        support_points,
        labels,
        domains,
        sinkhorn_loss
):

    if sinkhorn_loss is None:
        raise ValueError(
            "sinkhorn_loss must be provided when "
            "mode='sinkhorn'."
        )

    if support_points.dim() != 3:
        raise ValueError(
            "Sinkhorn mode requires support_points with shape "
            "[num_classes, num_supports, embedding_dim]."
        )

    num_classes = support_points.size(0)

    unique_domains = torch.unique(
        domains,
        sorted=True
    )

    losses = []

    for domain_id in unique_domains:

        for class_id in range(num_classes):

            mask = (
                (domains == domain_id)
                & (labels == class_id)
            )

            if not torch.any(mask):
                continue

            domain_features = features[mask]
            class_supports = support_points[class_id]

            # Wasserstein distance between:
            #   empirical domain-class distribution
            # and
            #   class support-point distribution
            distance = sinkhorn_loss(
                domain_features,
                class_supports
            )

            losses.append(distance)

    if len(losses) == 0:
        return features.new_zeros(
            (),
            requires_grad=True
        )

    return torch.stack(losses).mean()


def wass_center_loss(
        features,
        support_points,
        labels,
        domains,
        mode="sinkhorn",
        sinkhorn_loss=None
):

    mode = mode.lower()

    if mode == "sinkhorn":

        return sinkhorn_wass_loss(
            features=features,
            support_points=support_points,
            labels=labels,
            domains=domains,
            sinkhorn_loss=sinkhorn_loss
        )

    elif mode == "prototype":

        return prototype_wass_loss(
            features=features,
            support_points=support_points,
            labels=labels,
            domains=domains
        )

    else:
        raise ValueError(
            f"Unsupported Wasserstein mode: {mode}. "
            f"Choose from ['sinkhorn', 'prototype']."
        )