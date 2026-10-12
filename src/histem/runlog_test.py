from pathlib import Path

from matplotlib.figure import Figure

from histem.runlog import RunLog


def test_writes_an_event_file(tmp_path: Path) -> None:
    log = RunLog("t", root=tmp_path)
    log.scalar("loss", 1.0, 0)
    log.scalars("eval", {"model": 1.0, "control": 2.0}, 0)
    log.figure("fig", Figure(), 0)
    log.text("config", "x")
    log.close()
    assert list(log.dir.glob("events.out.tfevents.*"))
