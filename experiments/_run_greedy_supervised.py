"""Supervise _greedy_assemble.py; restart once on abnormal exit."""

# repo-root bootstrap: experiments/ scripts import strategy modules from the root
import sys as _sys
from pathlib import Path as _Path
_REPO_ROOT = _Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_REPO_ROOT))

import subprocess
import sys
import time

CMD = [sys.executable, "-u", "experiments/_greedy_assemble.py"]
MAX_ATTEMPTS = 2

for attempt in range(1, MAX_ATTEMPTS + 1):
    print(f"[supervisor] attempt {attempt}/{MAX_ATTEMPTS} starting", flush=True)
    t0 = time.time()
    p = subprocess.run(CMD)
    print(f"[supervisor] exited code={p.returncode} after {time.time()-t0:.0f}s",
          flush=True)
    if p.returncode == 0:
        print("[supervisor] GREEDY COMPLETE", flush=True)
        sys.exit(0)
    if attempt < MAX_ATTEMPTS:
        time.sleep(5)

sys.exit(p.returncode)
