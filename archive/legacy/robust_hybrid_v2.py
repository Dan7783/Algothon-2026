"""Conservative adaptive-atlas implementation for ``robust_hybrid``.

The non-atlas sleeves are intentionally identical to the selected hybrid.  The
production update changes only selected adaptive pair agents:

* MTNS/ALUT uses persistent disagreement and a capped, damped beta snap;
* ALUT/CCNS keeps its entry beta while a trade is open;
* MDGI/HTRK and DIHO/RTTH exit on cuts and cool down after adaptation;
* NAYO/SRTX now uses its robust 0.35 seed entry directly; no hidden floor.

Module-level controls exist for ablation research.  ``reset()`` constructs a
fresh atlas from the current controls, making independent replays deterministic.
"""

from __future__ import annotations

import numpy as np

import archive.old_tuners.daniel_filter as _daniel


N_INST = 51
LEG = 10_000.0

# Structural-beta controls.
OVERRIDE_FILTER_CONTROLS = True
CUT_CONFIRM_BARS = 3
CUT_COOLDOWN_BARS = 5
SNAP_WINDOW = 40
SNAP_OVERSHOOT = 1.0
SNAP_MAX_REL = 0.03
ADAPT_CHASE = 0.15
ADAPT_Q_MULT = 50.0
ADAPT_Q_FLOOR = 1.0e-6
ADAPT_Q_CAP = 1.0e-4
ADAPT_P_FLOOR = 1.0e-3
STAND_DOWN_WHILE_ADAPTING = False
FREEZE_BETA_WHILE_OPEN = False
EXIT_ON_CUT = False
FILTER_CONTROL_PAIR_NAMES: frozenset[str] | None = frozenset({"MTNS/ALUT"})
FREEZE_PAIR_NAMES: frozenset[str] | None = frozenset({"ALUT/CCNS"})
EXIT_ON_CUT_PAIR_NAMES: frozenset[str] | None = frozenset(
    {"MDGI/HTRK", "DIHO/RTTH"}
)
STAND_DOWN_PAIR_NAMES: frozenset[str] | None = EXIT_ON_CUT_PAIR_NAMES
CUT_COOLDOWN_PAIR_NAMES: frozenset[str] | None = EXIT_ON_CUT_PAIR_NAMES

# Execution controls.
ENTRY_FLOOR = 0.0
ENTRY_CONFIRM_BARS = 1
REQUIRE_TURN_CONFIRMATION = False
STAGED_ENTRY = False
ENTRY_SIZE = 1.0
FULL_SIZE_Z = 1.25
ALLOW_DIRECT_FLIP = True
ENTRY_FLOOR_BY_PAIR: dict[str, float] | None = None
ENTRY_CONFIRM_PAIR_NAMES: frozenset[str] | None = None
STAGED_ENTRY_PAIR_NAMES: frozenset[str] | None = None
TURN_CONFIRM_PAIR_NAMES: frozenset[str] | None = None

# The already-validated complementary sleeves remain unchanged.
USE_EXTRA_LL = True
USE_RIDGE = True
USE_DISCOVERY = True
RIDGE_ENTRY = 1.0
DISCOVERY_SCALE = 1.0
DISCOVERY_MIN_HISTORY = 50
DISCOVERY_FIT_WINDOW = 250
DISCOVERY_ENTRY_Z = 0.2
DISCOVERY_BREAK_Z = 4.5

# Reproducible production policy selected by full-window and four-half-window
# guards.  Research scripts may mutate module controls, then restore this map.
SELECTED_POLICY: dict[str, object] = {
    "OVERRIDE_FILTER_CONTROLS": True,
    "CUT_CONFIRM_BARS": 3,
    "CUT_COOLDOWN_BARS": 5,
    "SNAP_WINDOW": 40,
    "SNAP_OVERSHOOT": 1.0,
    "SNAP_MAX_REL": 0.03,
    "ADAPT_CHASE": 0.15,
    "ADAPT_Q_MULT": 50.0,
    "ADAPT_Q_FLOOR": 1.0e-6,
    "ADAPT_Q_CAP": 1.0e-4,
    "ADAPT_P_FLOOR": 1.0e-3,
    "STAND_DOWN_WHILE_ADAPTING": False,
    "FREEZE_BETA_WHILE_OPEN": False,
    "EXIT_ON_CUT": False,
    "FILTER_CONTROL_PAIR_NAMES": frozenset({"MTNS/ALUT"}),
    "FREEZE_PAIR_NAMES": frozenset({"ALUT/CCNS"}),
    "EXIT_ON_CUT_PAIR_NAMES": frozenset({"MDGI/HTRK", "DIHO/RTTH"}),
    "STAND_DOWN_PAIR_NAMES": frozenset({"MDGI/HTRK", "DIHO/RTTH"}),
    "CUT_COOLDOWN_PAIR_NAMES": frozenset({"MDGI/HTRK", "DIHO/RTTH"}),
    "ENTRY_FLOOR": 0.0,
    "ENTRY_CONFIRM_BARS": 1,
    "REQUIRE_TURN_CONFIRMATION": False,
    "STAGED_ENTRY": False,
    "ENTRY_SIZE": 1.0,
    "FULL_SIZE_Z": 1.25,
    "ALLOW_DIRECT_FLIP": True,
    "ENTRY_FLOOR_BY_PAIR": None,
    "ENTRY_CONFIRM_PAIR_NAMES": None,
    "STAGED_ENTRY_PAIR_NAMES": None,
    "TURN_CONFIRM_PAIR_NAMES": None,
}


def apply_policy(policy: dict[str, object]) -> None:
    """Apply a complete research/production policy and clear runtime state."""
    for name, value in policy.items():
        if name not in globals():
            raise KeyError(f"unknown policy control: {name}")
        globals()[name] = value
    reset()


def _discovery_universe() -> tuple[np.ndarray, np.ndarray]:
    atlas_keys = {
        tuple(sorted((int(spec.y), int(spec.x))))
        for spec in _daniel.PAIR_CFGS
    }
    ti, tj = np.triu_indices(N_INST - 1, k=1)
    pair_i, pair_j = ti + 1, tj + 1
    keep = np.fromiter(
        (
            tuple(sorted((int(i), int(j)))) not in atlas_keys
            for i, j in zip(pair_i, pair_j)
        ),
        dtype=bool,
        count=len(pair_i),
    )
    return pair_i[keep], pair_j[keep]


DISCOVERY_I, DISCOVERY_J = _discovery_universe()


def _capacity(existing: float, requested: float) -> float:
    if requested > 0.0:
        return max(0.0, (LEG - existing) / requested)
    if requested < 0.0:
        return max(0.0, (-LEG - existing) / requested)
    return 0.0


def _discovery_dollars(
    log_prices: np.ndarray,
    base_dollars: np.ndarray,
) -> np.ndarray:
    """Four Sigmas trailing-OLS discovery on residual idle capacity."""
    pos = np.zeros(log_prices.shape[0])
    if (
        not USE_DISCOVERY
        or DISCOVERY_SCALE <= 0.0
        or log_prices.shape[1] < DISCOVERY_MIN_HISTORY + 1
    ):
        return pos

    window = log_prices
    if log_prices.shape[1] > DISCOVERY_FIT_WINDOW + 1:
        window = log_prices[:, -(DISCOVERY_FIT_WINDOW + 1) :]
    train = window[:, :-1]
    y = train[DISCOVERY_I]
    x = train[DISCOVERY_J]
    yc = y - y.mean(axis=1, keepdims=True)
    xc = x - x.mean(axis=1, keepdims=True)
    beta = (yc * xc).sum(axis=1) / np.maximum((xc * xc).sum(axis=1), 1e-18)
    alpha = y.mean(axis=1) - beta * x.mean(axis=1)
    residual = y - alpha[:, None] - beta[:, None] * x
    current = (
        log_prices[DISCOVERY_I, -1]
        - alpha
        - beta * log_prices[DISCOVERY_J, -1]
    )
    z = current / (residual.std(axis=1) + 1e-12)
    idle = np.abs(base_dollars) <= 1.0
    eligible = (
        (beta > 0.0)
        & np.isfinite(beta)
        & np.isfinite(z)
        & (np.abs(z) >= DISCOVERY_ENTRY_Z)
        & (np.abs(z) < DISCOVERY_BREAK_Z)
        & idle[DISCOVERY_I]
        & idle[DISCOVERY_J]
    )

    used = np.zeros(log_prices.shape[0], dtype=bool)
    for k in np.argsort(-np.abs(z)):
        if not eligible[k]:
            continue
        i, j = int(DISCOVERY_I[k]), int(DISCOVERY_J[k])
        if used[i] or used[j]:
            continue
        direction = -np.sign(z[k])
        leg = LEG / max(1.0, abs(float(beta[k])))
        raw_i = direction * leg
        raw_j = -direction * float(beta[k]) * leg
        scale = min(
            DISCOVERY_SCALE,
            _capacity(float(base_dollars[i] + pos[i]), float(raw_i)),
            _capacity(float(base_dollars[j] + pos[j]), float(raw_j)),
        )
        if scale <= 1e-12:
            continue
        pos[i] += scale * raw_i
        pos[j] += scale * raw_j
        used[i] = used[j] = True
    return pos


def _candidate_filter(spec: _daniel.PairCfg) -> _daniel.FiltCfg:
    base = spec.filt
    use_controls = (
        OVERRIDE_FILTER_CONTROLS
        if FILTER_CONTROL_PAIR_NAMES is None
        else spec.name in FILTER_CONTROL_PAIR_NAMES
    )
    if not use_controls:
        return base
    q_adapt = min(
        max(base.q_beta * ADAPT_Q_MULT, ADAPT_Q_FLOOR),
        ADAPT_Q_CAP,
    )
    q_adapt = max(base.q_beta, q_adapt)
    return base._replace(
        q_beta_adapt=float(q_adapt),
        snap_window=(
            max(int(SNAP_WINDOW), 12)
            if SNAP_WINDOW > 0
            else int(base.snap_window)
        ),
        snap_overshoot=float(SNAP_OVERSHOOT),
        adapt_chase=float(ADAPT_CHASE),
        cut_confirm_bars=max(int(CUT_CONFIRM_BARS), 1),
        adapt_p_floor=max(float(ADAPT_P_FLOOR), 0.0),
        snap_max_rel=max(float(SNAP_MAX_REL), 0.0),
    )


class ConservativePairAgent:
    """Pair state with a shadow filter and a frozen execution hedge."""

    __slots__ = (
        "spec", "filt", "beta", "beta_lag", "trade_beta", "adapting",
        "current_cut", "cooldown", "nt_seen", "side", "window_z",
        "previous_z", "pending_side", "pending_count", "age",
    )

    def __init__(self, spec: _daniel.PairCfg) -> None:
        self.spec = spec
        self.filt = _daniel.PairSegFilter(_candidate_filter(spec))
        self.beta = 1.0
        self.beta_lag = 1.0
        self.trade_beta: float | None = None
        self.adapting = False
        self.current_cut = False
        self.cooldown = 0
        self.nt_seen = 0
        self.side = 0
        self.window_z = 0.0
        self.previous_z: float | None = None
        self.pending_side = 0
        self.pending_count = 0
        self.age = 0

    def _flatten(self) -> None:
        self.side = 0
        self.trade_beta = None
        self.pending_side = 0
        self.pending_count = 0
        self.age = 0

    def catch_up(self, prices: np.ndarray) -> None:
        nt = prices.shape[1]
        if self.nt_seen > nt:
            self.__init__(self.spec)
        y_i, x_i = self.spec.y, self.spec.x
        self.current_cut = False
        for t in range(self.nt_seen, nt):
            if self.cooldown > 0:
                self.cooldown -= 1
            self.beta_lag = self.beta
            y = np.log(max(float(prices[y_i, t]), 1e-12))
            x = np.log(max(float(prices[x_i, t]), 1e-12))
            output = self.filt.step(y, x)
            self.beta = float(output.beta)
            self.adapting = bool(output.adapting)
            self.current_cut = bool(output.cut)
            use_cooldown = (
                True
                if CUT_COOLDOWN_PAIR_NAMES is None
                else self.spec.name in CUT_COOLDOWN_PAIR_NAMES
            )
            if self.current_cut and use_cooldown:
                self.cooldown = max(self.cooldown, int(CUT_COOLDOWN_BARS))
            self.nt_seen = t + 1

    def _trade_scale(self, absolute_z: float, entry: float) -> float:
        staged = (
            STAGED_ENTRY
            if STAGED_ENTRY_PAIR_NAMES is None
            else self.spec.name in STAGED_ENTRY_PAIR_NAMES
        )
        if not staged:
            return 1.0
        initial = max(0.0, min(1.0, float(ENTRY_SIZE)))
        full_z = max(float(FULL_SIZE_Z), entry + 1e-9)
        progress = max(0.0, min(1.0, (absolute_z - entry) / (full_z - entry)))
        return initial + (1.0 - initial) * progress

    def desired_side(self, log_prices: np.ndarray) -> float:
        trade = self.spec.trade
        exit_on_cut = (
            EXIT_ON_CUT
            if EXIT_ON_CUT_PAIR_NAMES is None
            else self.spec.name in EXIT_ON_CUT_PAIR_NAMES
        )
        stand_down = (
            STAND_DOWN_WHILE_ADAPTING
            if STAND_DOWN_PAIR_NAMES is None
            else self.spec.name in STAND_DOWN_PAIR_NAMES
        )
        freeze_beta = (
            FREEZE_BETA_WHILE_OPEN
            if FREEZE_PAIR_NAMES is None
            else self.spec.name in FREEZE_PAIR_NAMES
        )
        if exit_on_cut and self.current_cut:
            self._flatten()
            return 0.0
        if stand_down and self.adapting:
            self._flatten()
            return 0.0

        nt = log_prices.shape[1]
        if nt < _daniel.HEDGE_W + 10:
            self._flatten()
            return 0.0

        y_i, x_i = self.spec.y, self.spec.x
        beta = (
            float(self.trade_beta)
            if freeze_beta and self.side and self.trade_beta is not None
            else float(self.beta_lag)
        )
        if trade.require_adf:
            spread = log_prices[y_i, : nt - 1] - beta * log_prices[x_i, : nt - 1]
            if (
                len(spread) >= 40
                and _daniel._adf_t_spread(spread) >= _daniel.PRIOR_GATE_T
            ):
                self._flatten()
                return 0.0

        spread = (
            log_prices[y_i, nt - _daniel.Z_W : nt]
            - beta * log_prices[x_i, nt - _daniel.Z_W : nt]
        )
        z = float(
            (spread[-1] - spread[:-1].mean())
            / (spread[:-1].std() + 1e-12)
        )
        self.window_z = z
        if trade.require_kappa:
            spread_120 = (
                log_prices[y_i, nt - 120 : nt]
                - beta * log_prices[x_i, nt - 120 : nt]
            )
            if _daniel._ou_kappa(spread_120) <= 0.0:
                self._flatten()
                self.previous_z = z
                return 0.0

        absolute_z = abs(z)
        floor = (
            float(ENTRY_FLOOR)
            if ENTRY_FLOOR_BY_PAIR is None
            else float(ENTRY_FLOOR_BY_PAIR.get(self.spec.name, 0.0))
        )
        entry = max(float(trade.z_entry), floor)
        exit_z = min(float(trade.z_exit), 0.85 * entry)
        stop = max(float(trade.z_stop), 1.25 * entry)
        desired = int(-np.sign(z))

        if self.side == 0:
            can_enter = self.cooldown <= 0 and entry <= absolute_z < stop
            turn_confirmation = (
                REQUIRE_TURN_CONFIRMATION
                if TURN_CONFIRM_PAIR_NAMES is None
                else self.spec.name in TURN_CONFIRM_PAIR_NAMES
            )
            if can_enter and turn_confirmation:
                can_enter = (
                    self.previous_z is not None
                    and desired * (z - self.previous_z) > 0.0
                )
            if can_enter:
                if desired == self.pending_side:
                    self.pending_count += 1
                else:
                    self.pending_side = desired
                    self.pending_count = 1
                confirm_bars = (
                    max(int(ENTRY_CONFIRM_BARS), 1)
                    if ENTRY_CONFIRM_PAIR_NAMES is None
                    else (
                        max(int(ENTRY_CONFIRM_BARS), 1)
                        if self.spec.name in ENTRY_CONFIRM_PAIR_NAMES
                        else 1
                    )
                )
                if self.pending_count >= confirm_bars:
                    self.side = desired
                    self.trade_beta = beta
                    self.age = 1
                    self.pending_side = 0
                    self.pending_count = 0
            else:
                self.pending_side = 0
                self.pending_count = 0
        else:
            self.age += 1
            if absolute_z <= exit_z or absolute_z >= stop:
                self._flatten()
            elif desired != self.side and absolute_z >= entry:
                if ALLOW_DIRECT_FLIP:
                    self.side = desired
                    self.trade_beta = beta
                    self.age = 1
                else:
                    self._flatten()
                    self.cooldown = max(self.cooldown, 1)

        self.previous_z = z
        if not self.side:
            return 0.0
        return float(self.side) * self._trade_scale(absolute_z, entry)


class ConservativeAtlasBook:
    """Daniel's EV-MWIS allocator driven by conservative pair agents."""

    def __init__(self) -> None:
        self.agents = [ConservativePairAgent(spec) for spec in _daniel.PAIR_CFGS]
        self.active: set[int] = set()

    def dollars(self, prices: np.ndarray) -> np.ndarray:
        nins, nt = prices.shape
        pos = np.zeros(nins)
        if nt < _daniel.HEDGE_W + 10:
            self.active = set()
            return pos

        log_prices = np.log(np.maximum(prices, 1e-12))
        candidates: list[tuple[float, int, float]] = []
        for agent in self.agents:
            agent.catch_up(prices)
            signal = agent.desired_side(log_prices)
            if abs(signal) < 1e-12:
                continue
            spec = agent.spec
            hub = max(_daniel.HUB_PEN.get(spec.idx, 1.0), 1.0)
            ev = (
                spec.priority
                * abs(agent.window_z)
                * spec.trade.leg_scale
                * abs(signal)
                / hub
            )
            candidates.append((ev, spec.idx, signal))

        candidates.sort(reverse=True)
        taken: set[int] = set()
        selected: list[tuple[int, float]] = []
        for _ev, idx, signal in candidates:
            agent = self.agents[idx]
            y_i, x_i = agent.spec.y, agent.spec.x
            if y_i in taken or x_i in taken:
                continue
            taken.add(y_i)
            taken.add(x_i)
            selected.append((idx, signal))

        self.active = {idx for idx, _signal in selected}
        for idx, signal in selected:
            agent = self.agents[idx]
            scale = agent.spec.trade.leg_scale
            pos[agent.spec.y] += signal * LEG * scale
            pos[agent.spec.x] -= signal * LEG * scale
        return pos


_STATE: dict[str, ConservativeAtlasBook | None] = {"atlas": None}


def reset() -> None:
    _STATE["atlas"] = None
    _daniel.RIDGE_THETA = float(RIDGE_ENTRY)


def sleeve_breakdown(prcSoFar: np.ndarray) -> dict[str, np.ndarray] | None:
    nins, nt = prcSoFar.shape
    if nins != N_INST or nt < 130:
        return None
    recent = prcSoFar[:, -3:]
    if not np.all(np.isfinite(recent)) or np.any(recent <= 0.0):
        return None
    if _STATE["atlas"] is None:
        _STATE["atlas"] = ConservativeAtlasBook()

    log_prices = np.log(prcSoFar)
    returns = np.diff(log_prices, axis=1)
    atlas = _STATE["atlas"].dollars(prcSoFar)

    extra_idx = (
        _daniel._EXTRA_IDX_NO_PAIR_FOLLOWERS
        if _daniel.FILTER_EXTRA_PAIR_FOLLOWERS
        else _daniel._EXTRA_IDX
    )
    extra = (
        _daniel._extra_leadlag_dollars(
            returns, extra_idx, np.abs(atlas) > 1.0
        )
        if USE_EXTRA_LL
        else np.zeros(nins)
    )
    if _daniel.PAIR_PRIORITY_OVER_LL:
        extra = np.where(np.abs(atlas) > 1.0, 0.0, extra)

    _daniel.RIDGE_THETA = float(RIDGE_ENTRY)
    ridge = (
        _daniel._ridge_dollars(returns, atlas + extra)
        if USE_RIDGE
        else np.zeros(nins)
    )
    held = atlas + extra + ridge
    held[1:] = np.clip(held[1:], -LEG, LEG)

    discovery = _discovery_dollars(log_prices, held)
    return {
        "atlas": atlas,
        "extraLL": extra,
        "ridge": ridge,
        "discovery": discovery,
    }


def getMyPosition(prcSoFar: np.ndarray) -> np.ndarray:
    nins, nt = prcSoFar.shape
    if nins != N_INST or nt < 130:
        return np.zeros(nins, dtype=int)
    recent = prcSoFar[:, -3:]
    if not np.all(np.isfinite(recent)) or np.any(recent <= 0.0):
        return np.zeros(nins, dtype=int)

    sleeves = sleeve_breakdown(prcSoFar)
    if sleeves is None:
        return np.zeros(nins, dtype=int)
    dollars = sum(sleeves.values(), start=np.zeros(nins))

    last_return = np.diff(np.log(prcSoFar[:, -2:]), axis=1)[:, 0]
    crash = np.abs(last_return) > 0.40
    crash[0] = False
    dollars[crash] = 0.0
    dollars[1:] = np.clip(dollars[1:], -LEG, LEG)
    dollars[0] = 0.0
    return (dollars / prcSoFar[:, -1]).astype(int)
