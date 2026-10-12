"""TensorBoard run logging. View with `make tb`."""

from datetime import datetime
from pathlib import Path

from matplotlib.figure import Figure
from torch.utils.tensorboard import SummaryWriter

from histem.data import DATA_DIR

# ../runs next to the repo
RUNS_DIR = DATA_DIR.parent / "runs"


class RunLog:
    def __init__(self, name: str, root: Path = RUNS_DIR):
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        self.dir = root / f"{stamp}-{name}"
        self._writer = SummaryWriter(self.dir)

    def scalar(self, tag: str, value: float, step: int) -> None:
        self._writer.add_scalar(tag, value, step)

    def scalars(self, tag: str, values: dict[str, float], step: int) -> None:
        """Several lines on one chart."""
        self._writer.add_scalars(tag, values, step)

    def figure(self, tag: str, fig: Figure, step: int) -> None:
        self._writer.add_figure(tag, fig, step, close=True)

    def text(self, tag: str, text: str, step: int = 0) -> None:
        self._writer.add_text(tag, text, step)

    def flush(self) -> None:
        self._writer.flush()

    def close(self) -> None:
        self._writer.close()
