"""Clean re-eval of the cherry-picked catalog (margin 0.30, all cohorts) —
earlier cherry numbers ran with leaked env (margin 0.50 + reject excluded).
Uses the current merged_strategy.py embedded catalog (still the cherry splice)
with explicitly pinned env.
"""

# repo-root bootstrap: experiments/ scripts import strategy modules from the root
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

import os
import re
import subprocess
import sys

env = os.environ.copy()
env["MERGED_CONSENSUS_MARGIN"] = "0.30"
env["MERGED_PROTECTED_BOOST"] = "4.0"
env.pop("MERGED_EXCLUDE_COHORTS", None)
env.pop("MERGED_EXCLUDE_PAIRS", None)
env.pop("MERGED_CATALOG_JSON", None)

for label, start, end in (("OOS", "801", "1000"), ("full", "200", "1000")):
    p = subprocess.run(
        [sys.executable, "eval.py", "--module", "merged_strategy",
         "--start", start, "--end", end],
        capture_output=True, text=True, env=env,
    )
    m = re.search(r"Score: ([\d.]+)", p.stdout)
    sh = re.search(r"annSharpe\(PL\): ([\d.]+)", p.stdout)
    print(f"cherry {label}: score={m.group(1) if m else '?'} "
          f"sharpe={sh.group(1) if sh else '?'}", flush=True)
print("CHERRY CLEAN DONE")
