import torch
from matplotlib.figure import Figure


def shift_scatter(
    predicted: dict[str, torch.Tensor], observed: torch.Tensor, title: str
) -> Figure:
    """Per-gene mean shift from control: observed on x, each prediction on y."""
    fig = Figure(figsize=(4.5, 4.5))
    ax = fig.add_subplot()
    lim = float(observed.abs().max().clamp(min=0.1)) * 1.1
    ax.plot([-lim, lim], [-lim, lim], color="0.7", lw=1)
    for name, pred in predicted.items():
        r = torch.corrcoef(torch.stack([observed, pred]))[0, 1]
        ax.scatter(observed, pred, s=6, alpha=0.6, label=f"{name} (r={r:.2f})")
    ax.set(xlabel="observed shift", ylabel="predicted shift", title=title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig


def pca_overlay(
    real: torch.Tensor, sim: torch.Tensor, control: torch.Tensor, title: str
) -> Figure:
    """Real, simulated and control cells on the top two PCs of the real cells and
    control together."""
    ref = torch.cat([real, control])
    mu = ref.mean(dim=0)
    _, _, v = torch.pca_lowrank(ref - mu, q=2, center=False)
    fig = Figure(figsize=(4.5, 4.5))
    ax = fig.add_subplot()
    for name, x, color in [
        ("control", control, "0.75"),
        ("real", real, "tab:blue"),
        ("simulated", sim, "tab:orange"),
    ]:
        pc = (x - mu) @ v
        ax.scatter(pc[:, 0], pc[:, 1], s=4, alpha=0.5, color=color, label=name)
    ax.set(xlabel="PC1", ylabel="PC2", title=title)
    ax.legend(fontsize=7, markerscale=3)
    fig.tight_layout()
    return fig


def condition_bars(scores: dict[str, dict[str, float]], title: str) -> Figure:
    """scores[condition][method] as grouped bars, conditions sorted by the first
    method."""
    conditions = sorted(scores, key=lambda c: next(iter(scores[c].values())))
    methods = list(next(iter(scores.values())))
    fig = Figure(figsize=(max(6.0, 0.3 * len(conditions)), 4))
    ax = fig.add_subplot()
    w = 0.8 / len(methods)
    for i, m in enumerate(methods):
        xs = [j + i * w for j in range(len(conditions))]
        ax.bar(xs, [scores[c][m] for c in conditions], w, label=m)
    ax.set_xticks([j + 0.4 - w / 2 for j in range(len(conditions))])
    ax.set_xticklabels(conditions, rotation=90, fontsize=6)
    ax.set(ylabel="energy distance", title=title)
    ax.legend(fontsize=7)
    fig.tight_layout()
    return fig
