# histem

Induce **executable, shared-rule cell dynamics** from data. Every cell runs the same rules
(the "genome") from its own state (expression, chromatin, mitochondria, latent slots).
Fitting means finding rules whose simulated cells look like the measured ones.
A fitted world can instantiate a patch of cells and run it forward.

Experimental. Highly unstable.

## Pieces

Code lives in `src/histem/`. Unit tests sit next to the module they test
(`foo.py` → `foo_test.py`); `tests/` holds only integration tests.

| module | role |
|---|---|
| `state.py` | `StateSchema` of typed `Slot`s; `Population` = batched per-cell state |
| `dynamics.py` | `Dynamics` protocol: the pluggable rules, a stochastic transition kernel `step(state, inputs) -> state`. Also `Intervention` (clamps + tags). |
| `simulator.py` | runs a population forward; routes auto/para/endocrine signals over an optional neighbor graph |
| `observers.py` | per-modality observation models (state -> scRNA counts, ...). New data types are new Observers. |
| `world.py` | `WorldModel` = dynamics + init prior + signaling + observers |
| `data.py`, `suite.py`, `metrics.py` | datasets as a regression suite; objective = distributional fit (energy distance) + λ·description length |
| `models/logic.py` | first `Dynamics`: multi-valued asynchronous logic programs written as plain text |
| `learners/search.py` | `Proposer` seam + hill climbing; `RandomLogicEdit` baseline |
| `synthetic.py` | hand-written ground-truth world for recovery experiments |

Any representation (LLM-written code, a conditional discrete-diffusion kernel, a
distilled neural emulator, ...) plugs in by implementing `Dynamics`.

## Usage

```bash
make install                  # uv sync (creates .venv, installs dev group)
make install-precommit-hooks  # ruff, ruff-format, mypy, file hygiene on commit
make check                    # ruff + format check, mypy, unit tests, integration tests
make format                   # ruff --fix + ruff format
uv run scripts/synthetic_recovery.py --iters 500
uv run scripts/fetch_data.py --list        # data goes to ../data
```
