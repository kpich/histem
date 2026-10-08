"""Download public datasets into ../data (a sibling of the repo).

    uv run scripts/fetch_data.py                 # default: norman2019
    uv run scripts/fetch_data.py replogle_k562_essential replogle_rpe1
    uv run scripts/fetch_data.py --list

Perturb-seq files come from the scPerturb harmonised h5ad collection on Zenodo; md5s
are checked against the Zenodo API.
"""

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

from histem.data import DATA_DIR

SCPERTURB_RECORD = "13350497"  # scPerturb RNA+protein h5ad, v1.4

DATASETS = {
    # name: (zenodo record, filename, note)
    "norman2019": (
        SCPERTURB_RECORD,
        "NormanWeissman2019_filtered.h5ad",
        "K562 CRISPRa, single + combinatorial; ~0.7 GB",
    ),
    "replogle_k562_essential": (
        SCPERTURB_RECORD,
        "ReplogleWeissman2022_K562_essential.h5ad",
        "K562 CRISPRi essential genes; ~1.5 GB",
    ),
    "replogle_rpe1": (
        SCPERTURB_RECORD,
        "ReplogleWeissman2022_rpe1.h5ad",
        "RPE1 CRISPRi essential genes; ~1.2 GB",
    ),
    "replogle_k562_gwps": (
        SCPERTURB_RECORD,
        "ReplogleWeissman2022_K562_gwps.h5ad",
        "K562 genome-wide CRISPRi; ~8.8 GB",
    ),
}


def zenodo_file(record: str, filename: str) -> tuple[str, str | None, int]:
    with urllib.request.urlopen(f"https://zenodo.org/api/records/{record}") as r:
        files = json.load(r)["files"]
    for f in files:
        if f["key"] == filename:
            md5 = (
                f["checksum"].split(":", 1)[1]
                if f.get("checksum", "").startswith("md5:")
                else None
            )
            return f["links"]["self"], md5, f["size"]
    raise FileNotFoundError(f"{filename} not in zenodo record {record}")


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, size: int) -> None:
    tmp = dest.with_suffix(dest.suffix + ".part")
    done = 0
    with urllib.request.urlopen(url) as r, tmp.open("wb") as f:
        while chunk := r.read(1 << 22):
            f.write(chunk)
            done += len(chunk)
            print(
                f"\r  {done / 1e9:.2f} / {size / 1e9:.2f} GB", end="", file=sys.stderr
            )
    print(file=sys.stderr)
    tmp.rename(dest)


def fetch(name: str, force: bool = False) -> Path:
    record, filename, _ = DATASETS[name]
    dest = DATA_DIR / "raw" / name / filename
    dest.parent.mkdir(parents=True, exist_ok=True)
    url, md5, size = zenodo_file(record, filename)
    if dest.exists() and not force:
        print(f"{name}: already at {dest}")
    else:
        print(f"{name}: downloading {filename} -> {dest}")
        download(url, dest, size)
    if md5 and md5sum(dest) != md5:
        raise RuntimeError(f"{name}: md5 mismatch for {dest}; delete it and re-run")
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=["norman2019"])
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    if args.list:
        for k, (_, fn, note) in DATASETS.items():
            print(f"{k:<26} {fn:<45} {note}")
        return
    for name in args.names:
        fetch(name, args.force)


if __name__ == "__main__":
    main()
