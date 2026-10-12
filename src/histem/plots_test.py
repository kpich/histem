import torch

from histem import plots


def test_figures_draw() -> None:
    torch.manual_seed(0)
    x = torch.randn(30, 5)
    figs = [
        plots.shift_scatter({"model": x[0], "additive": x[1]}, x[2], "t"),
        plots.pca_overlay(x, x + 1, x - 1, "t"),
        plots.condition_bars(
            {"a": {"m": 1.0, "b": 2.0}, "c": {"m": 0.5, "b": 1.0}}, "t"
        ),
    ]
    for fig in figs:
        fig.canvas.draw()
