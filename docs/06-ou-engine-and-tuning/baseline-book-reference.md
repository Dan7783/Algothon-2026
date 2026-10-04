# Baseline book (friend_book_latest.py) reference notes

> **Source:** `FRIEND_BOOK_REFERENCE.md` in the team's working repository (relocated here for the public release).

```text
# Friend book reference
#
# Canonical latest paste: friend_book_latest.py
# Previous snapshot kept for diffs: friend_book.py
#
# Material conflict-layer changes (latest vs old):
# - RIDGE_LAM/THETA retuned to 1.52/0.31; legacy 1.0/0.75 kept as RIDGE_CONFIRM_*
# - New _resolve_pair_ridge_direction: ridge may overturn opposing pair votes
#   only when confirm-ridge agrees and preliminary combined vote already prefers ridge
# - DEADLOCK_RIDGE_OVERLAY_FRACTION 0.25 -> 1.0 (full unused slot on deadlock)
# - USE_ALGO_RESIDUAL_BASKET remains True; comment updated to "part of current book"
#
# Hybrid strategy using this skeleton + OU winners: merged_friend_ou.py
# Rebuild: python _build_merged_friend_ou.py
```
