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


def probe_accuracy(
    cells: dict[str, torch.Tensor], steps: int = 100, weight_decay: float = 1e-3
) -> float:
    """Accuracy of a linear classifier for which group each cell came from, fit on a
    random half of every group's cells and scored on the other half."""
    xs, ys, test = [], [], []
    for label, x in enumerate(cells.values()):
        order = torch.randperm(x.shape[0])
        xs.append(x[order])
        ys.append(torch.full((x.shape[0],), label))
        test.append(torch.arange(x.shape[0]) >= x.shape[0] // 2)
    x, y, is_test = torch.cat(xs), torch.cat(ys), torch.cat(test)
    mu, sd = x[~is_test].mean(0), x[~is_test].std(0).clamp(min=1e-6)
    x = (x - mu) / sd
    w = torch.zeros(x.shape[1], len(cells), requires_grad=True)
    b = torch.zeros(len(cells), requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        opt.zero_grad()
        logits = x[~is_test] @ w + b
        loss = torch.nn.functional.cross_entropy(logits, y[~is_test])
        loss = loss + weight_decay * w.pow(2).sum()
        loss.backward()
        return loss

    with torch.enable_grad():
        opt.step(closure)
    with torch.no_grad():
        pred = (x[is_test] @ w + b).argmax(1)
    return float((pred == y[is_test]).float().mean())
