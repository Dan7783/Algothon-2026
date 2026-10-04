#!/usr/bin/env python
"""Post-tuning pipeline: archive db, regenerate merged catalog with tuned
profiles, splice the new _MERGED_PAIR_PARAMS literal into merged_strategy.py,
and smoke-test the import. Run after the catalog88 tuner completes."""

# repo-root bootstrap: experiments/ scripts import strategy modules from the root
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))


import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile

WORKSPACE_DB = "auto_tuning_results/optuna_catalog88.db"
TEMP_DB = os.path.join(tempfile.gettempdir(), "optuna_catalog88.db")
LITERAL_TXT = "auto_tuning_results/_merged_literal.txt"
STRATEGY = "merged_strategy.py"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hybrid", action="store_true",
                    help="zband catalog pairs + ou tuned cohorts (passed to generator)")
    ap.add_argument("--cherrypick", action="store_true",
                    help="per-pair engine split from attribution (passed to generator)")
    ap.add_argument("--skip-archive", action="store_true")
    args = ap.parse_args()

    # 1) archive the optuna db into the workspace for downstream tooling
    if not args.skip_archive and os.path.exists(TEMP_DB):
        shutil.copy(TEMP_DB, WORKSPACE_DB)
        print(f"db archived -> {WORKSPACE_DB}")

    # 2) regenerate the catalog + literal
    gen_cmd = [sys.executable, "experiments/_build_merged_catalog.py"]
    if args.hybrid:
        gen_cmd.append("--hybrid")
    if args.cherrypick:
        gen_cmd.append("--cherrypick")
    subprocess.run(gen_cmd, check=True, stdout=open(LITERAL_TXT, "w"))
    with open(LITERAL_TXT) as f:
        lines = f.readlines()
    print(lines[0].strip())
    start = next(i for i, ln in enumerate(lines)
                 if ln.startswith("_MERGED_PAIR_PARAMS = ("))
    literal = lines[start:]

    # 3) splice into merged_strategy.py
    with open(STRATEGY) as f:
        src = f.readlines()
    i0 = next(i for i, ln in enumerate(src)
              if ln.startswith("_MERGED_PAIR_PARAMS = ("))
    i1 = next(i for i in range(i0 + 1, len(src)) if src[i].rstrip() == ")")
    new_src = src[:i0] + literal + ["\n"] + src[i1 + 1:]
    with open(STRATEGY, "w") as f:
        f.writelines(new_src)
    print(f"spliced {len(literal) - 2} catalog rows into {STRATEGY}")

    # 4) smoke test
    out = subprocess.run(
        [sys.executable, "-c",
         "import merged_strategy as m; print('pairs:', len(m.PAIR_CFGS))"],
        check=True, capture_output=True, text=True,
    )
    print(out.stdout.strip())


if __name__ == "__main__":
    main()
