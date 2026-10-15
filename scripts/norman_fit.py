"""Fit a latent diffusion kernel to Norman 2019 and score it on held-out double
perturbations against the control and additive baselines. Logs to TensorBoard
(`make tb`)."""

import argparse
import json
import time

import torch

from histem import baselines, latent, norman, plots
from histem.data import CountsDataset, combo_split
from histem.learners.gradient import gradient_fit
from histem.metrics import energy_distance
from histem.rng import seeded
from histem.runlog import RunLog
from histem.suite import Suite
from histem.system import CellSystem


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="norman")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--cells", type=int, default=128)
    ap.add_argument("--conditions", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-cells", type=int, default=300)
    ap.add_argument("--train-eval", type=int, default=16)
    ap.add_argument("--plot-conditions", type=int, default=4)
    ap.add_argument("--genes", type=int, default=1000)
    ap.add_argument("--max-cells", type=int, default=500)
    ap.add_argument("--held-out-frac", type=float, default=0.2)
    ap.add_argument("--latents", type=int, default=16)
    ap.add_argument("--levels", type=int, default=8)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--gene-dim", type=int, default=128)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--burn-in", type=int, default=20)
    ap.add_argument("--grad-steps", type=int, default=10)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args()


class Evaluator:
    """Energy distance, mean-shift correlation and the shift of the perturbed genes
    themselves, on watched train conditions and held-out doubles, logged with the
    baselines' at every call."""

    def __init__(
        self,
        ds: CountsDataset,
        watched: list[str],
        held_out: list[str],
        log: RunLog,
        args: argparse.Namespace,
    ):
        self.ds, self.watched, self.held_out, self.log = ds, watched, held_out, log
        self.n = args.eval_cells
        self.seed = args.seed
        self.n_plots = args.plot_conditions
        with torch.no_grad(), seeded(self.seed):
            self.control_ed = {
                c: float(baselines.control(ds, c, self.n)) for c in watched + held_out
            }
            self.additive_ed = {
                c: float(baselines.additive(ds, c, self.n)) for c in held_out
            }
        self.perturbed = watched[1:] + held_out
        self.real_shift = {c: _shift(ds, c) for c in self.perturbed}
        self.additive_shift = {c: baselines.additive_shift(ds, c) for c in held_out}

    def __call__(self, system: CellSystem, step: int) -> None:
        ds, n, held = self.ds, self.n, self.held_out
        ed, sims, reals = {}, {}, {}
        with torch.no_grad(), seeded(self.seed):
            for c in self.watched + held:
                sims[c] = ds.simulated(system, c, n).cpu()
                reals[c] = ds.normalized(c, n)
                ed[c] = float(energy_distance(reals[c], sims[c]))
        control_mean = sims["control"].mean(dim=0)
        model_shift = {c: sims[c].mean(dim=0) - control_mean for c in self.perturbed}
        train = self.watched[1:]
        r = _r(model_shift, self.real_shift)
        model_target = _target_shift(ds, model_shift)
        real_target = _target_shift(ds, self.real_shift)

        self.log.scalars(
            "energy/held_out",
            {
                "model": _mean(ed, held),
                "control": _mean(self.control_ed, held),
                "additive": _mean(self.additive_ed, held),
            },
            step,
        )
        self.log.scalars(
            "energy/train",
            {
                "model": _mean(ed, self.watched),
                "control": _mean(self.control_ed, self.watched),
            },
            step,
        )
        self.log.scalars(
            "shift_r/held_out",
            {
                "model": _mean(r, held),
                "additive": _mean(_r(self.additive_shift, self.real_shift), held),
            },
            step,
        )
        self.log.scalar("shift_r/train", _mean(r, train), step)
        for split, conds in [("held_out", held), ("train", train)]:
            measured = [c for c in conds if c in real_target]
            self.log.scalars(
                f"target_shift/{split}",
                {
                    "model": _mean(model_target, measured),
                    "real": _mean(real_target, measured),
                },
                step,
            )
        bars = {
            c: {
                "model": ed[c],
                "additive": self.additive_ed[c],
                "control": self.control_ed[c],
            }
            for c in held
        }
        self.log.figure(
            "held_out/energy", plots.condition_bars(bars, "held-out doubles"), step
        )
        for c in held[: self.n_plots]:
            shifts = {"model": model_shift[c], "additive": self.additive_shift[c]}
            self.log.figure(
                f"shift/{c}", plots.shift_scatter(shifts, self.real_shift[c], c), step
            )
            with seeded(self.seed):
                pca = plots.pca_overlay(reals[c], sims[c], reals["control"], c)
            self.log.figure(f"pca/{c}", pca, step)
        self.log.flush()
        system.save(self.log.dir / "system")
        print(
            f"[{step}] held-out model={_mean(ed, held):.3f} "
            f"additive={_mean(self.additive_ed, held):.3f} "
            f"control={_mean(self.control_ed, held):.3f}  "
            f"train model={_mean(ed, self.watched):.3f}"
        )


def main() -> None:
    args = _args()
    torch.manual_seed(args.seed)
    ds = norman.load(args.genes, args.max_cells)
    held_out = combo_split(ds, args.held_out_frac, args.seed)
    train, _ = ds.split(held_out)
    fit_suite = Suite(
        datasets=[train],
        cells_per_condition=args.cells,
        loss_conditions=args.conditions,
    )
    others = [c for c in train.conditions if c != "control"]
    g = torch.Generator().manual_seed(args.seed)
    picked = torch.randperm(len(others), generator=g)[: args.train_eval].tolist()
    watched = ["control", *sorted(others[i] for i in picked)]

    system = latent.make_system(
        ds,
        latents=args.latents,
        levels=args.levels,
        width=args.width,
        layers=args.layers,
        gene_dim=args.gene_dim,
        tau=args.tau,
        burn_in=args.burn_in,
        grad_steps=args.grad_steps,
        device=args.device,
    )
    log = RunLog(args.name)
    log.text("config", f"```\n{json.dumps(vars(args), indent=2)}\n```")
    log.text("held_out", ", ".join(held_out))
    print(f"logging to {log.dir}")
    print(f"{len(train.conditions)} train conditions, {len(held_out)} held out")
    evaluate = Evaluator(ds, watched, held_out, log, args)

    last = time.perf_counter()

    def on_step(step: int, loss: float, grad_norms: dict[str, float]) -> None:
        nonlocal last
        log.scalar("train/loss", loss, step)
        log.scalar("train/grad_norm", grad_norms["total"], step)
        log.scalars("grad_norm", _group_norms(grad_norms), step)
        log.scalar("train/sec_per_step", time.perf_counter() - last, step)
        if step % args.eval_every == 0 or step == args.steps:
            evaluate(system, step)
        last = time.perf_counter()

    evaluate(system, 0)
    start = last = time.perf_counter()
    gradient_fit(system, fit_suite, args.steps, lr=args.lr, callback=on_step)
    print(f"fit in {time.perf_counter() - start:.0f}s")
    log.close()


def _mean(d: dict[str, float], conditions: list[str]) -> float:
    return sum(d[c] for c in conditions) / len(conditions)


def _group_norms(norms: dict[str, float]) -> dict[str, float]:
    """Pre-clip grad norms of the kernel, decoder, dispersion and gene table."""
    squares: dict[str, float] = {}
    for name, v in norms.items():
        if name == "total":
            continue
        if name.endswith("log_dispersion"):
            group = "dispersion"
        else:
            group = {"dynamics": "kernel", "observers": "decoder"}.get(
                name.split("/")[0], name.split("/")[0]
            )
        squares[group] = squares.get(group, 0.0) + v**2
    return {k: v**0.5 for k, v in squares.items()}


def _target_shift(
    ds: CountsDataset, shifts: dict[str, torch.Tensor]
) -> dict[str, float]:
    """Mean shift over each condition's measured perturbed genes."""
    where = {g: i for i, g in enumerate(ds.features)}
    out = {}
    for c, shift in shifts.items():
        idx = [where[g] for g in ds.interventions[c].genes if g in where]
        if idx:
            out[c] = float(shift[idx].mean())
    return out


def _shift(ds: CountsDataset, condition: str) -> torch.Tensor:
    return ds.normalized(condition).mean(dim=0) - ds.normalized("control").mean(dim=0)


def _r(
    pred: dict[str, torch.Tensor], real: dict[str, torch.Tensor]
) -> dict[str, float]:
    return {
        c: float(torch.corrcoef(torch.stack([real[c], pred[c]]))[0, 1]) for c in pred
    }


if __name__ == "__main__":
    main()
