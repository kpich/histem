"""Fit a latent diffusion kernel to one dataset and score it on held-out
perturbations against the control baseline and a shift baseline: Norman 2019 holds
out doubles (baseline: sum of the singles' shifts), Replogle 2022 holds out singles
(baseline: mean of the training shifts). A Replogle system spans both datasets'
genes, so `--warm` from a Norman checkpoint and a cold start fit the same model.
Logs to TensorBoard (`make tb`)."""

import argparse
import json
import time
from pathlib import Path

import torch

from histem import baselines, latent, norman, plots, replogle
from histem.data import CountsDataset, combo_split, single_split
from histem.learners.gradient import gradient_fit, param_group
from histem.metrics import energy_distance, probe_accuracy
from histem.rng import seeded
from histem.runlog import RunLog
from histem.suite import Suite
from histem.system import CellSystem


def _args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["norman", "replogle"], default="norman")
    ap.add_argument("--name")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--cells", type=int, default=128)
    ap.add_argument("--conditions", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-cells", type=int, default=300)
    ap.add_argument("--shift-cells", type=int, default=500)
    ap.add_argument("--probe-cells", type=int, default=100)
    ap.add_argument("--de-genes", type=int, default=20)
    ap.add_argument("--train-eval", type=int, default=16)
    ap.add_argument("--plot-conditions", type=int, default=4)
    ap.add_argument("--genes", type=int, default=1000)
    ap.add_argument("--max-cells", type=int, default=500)
    ap.add_argument("--held-out-frac", type=float)
    ap.add_argument(
        "--train-perturbations", type=int, help="train on only this many perturbations"
    )
    ap.add_argument("--latents", type=int, default=16)
    ap.add_argument("--levels", type=int, default=8)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--gene-dim", type=int, default=128)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--tau", type=float, default=2.0)
    ap.add_argument("--sim-steps", type=int, default=20)
    ap.add_argument("--grad-steps", type=int, default=10)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--load", type=Path, help="start from a saved system")
    ap.add_argument(
        "--warm", type=Path, help="copy weights from a saved system by name"
    )
    args = ap.parse_args()
    args.name = args.name or args.dataset
    if args.held_out_frac is None:
        args.held_out_frac = 0.2 if args.dataset == "norman" else 0.1
    return args


class Evaluator:
    """Energy distance (over all genes and over each condition's top DE genes),
    mean-shift correlation and the shift of the perturbed genes themselves, on
    watched train conditions and held-out perturbations, logged with the control
    baseline's and `baseline`'s (a predicted shift per held-out condition) at every
    call. With `probe`, also how well a linear probe tells the singles apart from
    simulated cells, against the same probe on real cells."""

    def __init__(
        self,
        ds: CountsDataset,
        watched: list[str],
        held_out: list[str],
        log: RunLog,
        args: argparse.Namespace,
        *,
        baseline: dict[str, torch.Tensor],
        probe: bool,
    ):
        self.ds, self.watched, self.held_out, self.log = ds, watched, held_out, log
        self.n = args.eval_cells
        self.n_shift = max(args.shift_cells, args.eval_cells)
        self.n_probe = args.probe_cells
        self.singles = sorted(ds.singles().values()) if probe else []
        self.seed = args.seed
        self.n_plots = args.plot_conditions
        self.n_bars = 30
        self.perturbed = watched[1:] + held_out
        self.real_shift = {c: _shift(ds, c) for c in self.perturbed}
        self.de = {
            c: self.real_shift[c].abs().topk(args.de_genes).indices.tolist()
            for c in self.perturbed
        }
        with torch.no_grad(), seeded(self.seed):
            self.control_ed = {
                c: float(baselines.control(ds, c, self.n)) for c in watched + held_out
            }
            self.baseline_ed = {
                c: float(baselines.shifted(ds, c, self.n, baseline[c]))
                for c in held_out
            }
            self.control_de = {
                c: float(baselines.control(ds, c, self.n, self.de[c]))
                for c in self.perturbed
            }
            self.baseline_de = {
                c: float(baselines.shifted(ds, c, self.n, baseline[c], self.de[c]))
                for c in held_out
            }
            if probe:
                real = {c: ds.normalized(c, self.n_probe) for c in self.singles}
                self.real_probe = probe_accuracy(real)
        self.baseline_shift = baseline

    def __call__(self, system: CellSystem, step: int) -> None:
        ds, n, held = self.ds, self.n, self.held_out
        ed, de, sims, reals = {}, {}, {}, {}
        with torch.no_grad(), seeded(self.seed):
            for c in self.watched + held:
                sims[c] = ds.simulated(system, c, self.n_shift).cpu()
                reals[c] = ds.normalized(c, n)
                ed[c] = float(energy_distance(reals[c], sims[c][:n]))
                if c in self.de:
                    cols = self.de[c]
                    de[c] = float(energy_distance(reals[c][:, cols], sims[c][:n, cols]))
            if self.singles:
                probe = probe_accuracy(
                    {
                        c: ds.simulated(system, c, self.n_probe).cpu()
                        for c in self.singles
                    }
                )
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
                "baseline": _mean(self.baseline_ed, held),
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
            "energy_de/held_out",
            {
                "model": _mean(de, held),
                "control": _mean(self.control_de, held),
                "baseline": _mean(self.baseline_de, held),
            },
            step,
        )
        self.log.scalars(
            "energy_de/train",
            {"model": _mean(de, train), "control": _mean(self.control_de, train)},
            step,
        )
        if self.singles:
            self.log.scalars(
                "probe/singles", {"model": probe, "real": self.real_probe}, step
            )
        self.log.scalars(
            "shift_r/held_out",
            {
                "model": _mean(r, held),
                "baseline": _mean(_r(self.baseline_shift, self.real_shift), held),
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
                "baseline": self.baseline_ed[c],
                "control": self.control_ed[c],
            }
            for c in held[: self.n_bars]
        }
        self.log.figure("held_out/energy", plots.condition_bars(bars, "held-out"), step)
        for c in held[: self.n_plots]:
            shifts = {"model": model_shift[c], "baseline": self.baseline_shift[c]}
            self.log.figure(
                f"shift/{c}", plots.shift_scatter(shifts, self.real_shift[c], c), step
            )
            with seeded(self.seed):
                pca = plots.pca_overlay(reals[c], sims[c][:n], reals["control"], c)
            self.log.figure(f"pca/{c}", pca, step)
        self.log.flush()
        system.save(self.log.dir / "system")
        print(
            f"[{step}] held-out model={_mean(ed, held):.3f} "
            f"baseline={_mean(self.baseline_ed, held):.3f} "
            f"control={_mean(self.control_ed, held):.3f}  "
            f"train model={_mean(ed, self.watched):.3f}"
            + (
                f"  probe model={probe:.3f} real={self.real_probe:.3f}"
                if self.singles
                else ""
            )
        )


def main() -> None:
    args = _args()
    torch.manual_seed(args.seed)
    ds, spanned, held_out = _data(args)
    train, _ = ds.split(held_out)
    if args.train_perturbations is not None:
        train = _subsample(train, args.train_perturbations, args.seed)
    perturbed = [c for c in train.conditions if c != "control"]
    if args.dataset == "norman":
        baseline = {c: baselines.additive_shift(ds, c) for c in held_out}
    else:
        baseline = dict.fromkeys(held_out, baselines.mean_shift(ds, perturbed))
    fit_suite = Suite(
        datasets=[train],
        cells_per_condition=args.cells,
        loss_conditions=args.conditions,
    )
    g = torch.Generator().manual_seed(args.seed)
    picked = torch.randperm(len(perturbed), generator=g)[: args.train_eval].tolist()
    watched = ["control", *sorted(perturbed[i] for i in picked)]

    if args.load:
        system = CellSystem.load(args.load).to(args.device)
    else:
        system = latent.make_system(
            spanned,
            latents=args.latents,
            levels=args.levels,
            width=args.width,
            layers=args.layers,
            gene_dim=args.gene_dim,
            rank=args.rank,
            tau=args.tau,
            steps=args.sim_steps,
            grad_steps=args.grad_steps,
            device=args.device,
        )
        if args.warm:
            fresh = system.load_weights(CellSystem.load(args.warm))
            print(f"warm start from {args.warm}; fresh: {', '.join(fresh)}")
    log = RunLog(args.name)
    log.text("config", f"```\n{json.dumps(vars(args), indent=2, default=str)}\n```")
    log.text("held_out", ", ".join(held_out))
    print(f"logging to {log.dir}")
    print(f"{len(train.conditions)} train conditions, {len(held_out)} held out")
    evaluate = Evaluator(
        ds,
        watched,
        held_out,
        log,
        args,
        baseline=baseline,
        probe=args.dataset == "norman",
    )

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
    gradient_fit(
        system,
        fit_suite,
        args.steps,
        lr=args.lr,
        group=param_group,
        callback=on_step,
    )
    print(f"fit in {time.perf_counter() - start:.0f}s")
    log.close()


def _data(
    args: argparse.Namespace,
) -> tuple[CountsDataset, list[CountsDataset], list[str]]:
    """The dataset to fit, every dataset the system spans, and the held-out
    conditions."""
    if args.dataset == "norman":
        ds = norman.load(args.genes, args.max_cells)
        return ds, [ds], combo_split(ds, args.held_out_frac, args.seed)
    ds = replogle.load(args.genes, args.max_cells)
    spanned = [norman.load(args.genes, args.max_cells), ds]
    return ds, spanned, single_split(ds, args.held_out_frac, args.seed)


def _subsample(ds: CountsDataset, n: int, seed: int) -> CountsDataset:
    """Control plus a seeded random n of the other conditions; a larger n keeps a
    superset."""
    others = [c for c in ds.conditions if c != "control"]
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(others), generator=g).tolist()
    dropped = [others[i] for i in order[n:]]
    kept, _ = ds.split(dropped)
    return kept


def _mean(d: dict[str, float], conditions: list[str]) -> float:
    return sum(d[c] for c in conditions) / len(conditions)


def _group_norms(norms: dict[str, float]) -> dict[str, float]:
    """Pre-clip grad norms of each mechanism, decoder, dispersion and gene table."""
    squares: dict[str, float] = {}
    for name, v in norms.items():
        if name == "total":
            continue
        group = param_group(name)
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
