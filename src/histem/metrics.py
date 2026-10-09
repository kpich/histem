import torch


def log_normalize(
    counts: torch.Tensor, totals: torch.Tensor, target: float = 1e4
) -> torch.Tensor:
    return torch.log1p(counts / totals.clamp(min=1).unsqueeze(1) * target)


def energy_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Energy distance (Székely); zero iff the distributions match."""
    return (
        2 * torch.cdist(x, y).mean()
        - torch.cdist(x, x).mean()
        - torch.cdist(y, y).mean()
    )
