"""Two-sided CUSUM detector on a standardized innovation stream.

Standalone and unit-testable against a plain numpy ``z_k`` array.
No dependency on Kalman filtering or regime-switching logic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# Named defaults (overridable per call)
DEFAULT_KAPPA: float = 0.25
DEFAULT_H: float = 9.0  # placeholder; prefer ARL-calibrated h
DEFAULT_EXIT_BAND: float = 0.5


@dataclass(frozen=True)
class CusumConfig:
    """CUSUM parameters.

    kappa:
        Allowance / reference value subtracted from |z| each step (default 0.5).
    h:
        Decision threshold; alarm when S+ > h or S- < -h (default 5.0;
        typically overwritten by ARL calibration).
    exit_band:
        Half-width around zero used by hysteresis exit logic (default 0.5).
        The detector itself does not enforce hysteresis — callers do.
    """

    kappa: float = DEFAULT_KAPPA
    h: float = DEFAULT_H
    exit_band: float = DEFAULT_EXIT_BAND

    def __post_init__(self) -> None:
        if self.kappa < 0.0:
            raise ValueError(f"kappa must be >= 0, got {self.kappa}")
        if self.h <= 0.0:
            raise ValueError(f"h must be > 0, got {self.h}")
        if self.exit_band < 0.0:
            raise ValueError(f"exit_band must be >= 0, got {self.exit_band}")


@dataclass(frozen=True)
class CusumState:
    """Mutable CUSUM recursion state (immutable snapshot)."""

    s_pos: float = 0.0
    s_neg: float = 0.0

    @property
    def alarm(self) -> bool:
        """Whether either side would alarm given an external h — unused here."""
        return False


@dataclass(frozen=True)
class CusumResult:
    """Full-path CUSUM output for a z series."""

    s_pos: np.ndarray
    s_neg: np.ndarray
    alarm: np.ndarray  # bool, True on timesteps where |CUSUM| breached h
    first_alarm_index: int | None
    config: CusumConfig


def cusum_step(
    z: float,
    state: CusumState,
    *,
    kappa: float = DEFAULT_KAPPA,
    h: float = DEFAULT_H,
) -> tuple[CusumState, bool]:
    """One CUSUM recursion step.

    Returns
    -------
    new_state, alarmed
    """
    s_pos = max(0.0, state.s_pos + float(z) - kappa)
    s_neg = min(0.0, state.s_neg + float(z) + kappa)
    alarmed = (s_pos > h) or (s_neg < -h)
    return CusumState(s_pos=s_pos, s_neg=s_neg), alarmed


def within_exit_band(
    state: CusumState,
    *,
    exit_band: float = DEFAULT_EXIT_BAND,
) -> bool:
    """True when both CUSUM sides lie inside [-exit_band, +exit_band]."""
    return (state.s_pos <= exit_band) and (state.s_neg >= -exit_band)


def run_cusum(
    z: np.ndarray,
    *,
    kappa: float = DEFAULT_KAPPA,
    h: float = DEFAULT_H,
    exit_band: float = DEFAULT_EXIT_BAND,
    reset_on_alarm: bool = False,
) -> CusumResult:
    """Run two-sided CUSUM over an entire standardized-innovation series.

    Parameters
    ----------
    z:
        1-d array of innovations (ideally ~N(0,1) under the null).
    kappa, h, exit_band:
        See ``CusumConfig`` (defaults 0.5, 5.0, 0.5).
    reset_on_alarm:
        If True, reset both sides to 0 after an alarm (useful for ARL
        run-length measurement of successive false alarms).  Default False
        keeps the statistic climbing after the first breach.
    """
    z = np.asarray(z, dtype=float).ravel()
    config = CusumConfig(kappa=kappa, h=h, exit_band=exit_band)

    T = z.size
    s_pos = np.empty(T, dtype=float)
    s_neg = np.empty(T, dtype=float)
    alarm = np.zeros(T, dtype=bool)

    state = CusumState()
    first_alarm: int | None = None

    for t in range(T):
        if not np.isfinite(z[t]):
            s_pos[t] = state.s_pos
            s_neg[t] = state.s_neg
            continue
        state, alarmed = cusum_step(z[t], state, kappa=kappa, h=h)
        s_pos[t] = state.s_pos
        s_neg[t] = state.s_neg
        alarm[t] = alarmed
        if alarmed:
            if first_alarm is None:
                first_alarm = t
            if reset_on_alarm:
                state = CusumState()

    return CusumResult(
        s_pos=s_pos,
        s_neg=s_neg,
        alarm=alarm,
        first_alarm_index=first_alarm,
        config=config,
    )


def run_length_to_alarm(
    z: np.ndarray,
    *,
    kappa: float = DEFAULT_KAPPA,
    h: float = DEFAULT_H,
) -> int:
    """Return observations until first alarm, or ``len(z)`` if none."""
    z = np.asarray(z, dtype=float).ravel()
    s_pos = 0.0
    s_neg = 0.0
    for t, zk in enumerate(z):
        if not np.isfinite(zk):
            continue
        s_pos = max(0.0, s_pos + zk - kappa)
        s_neg = min(0.0, s_neg + zk + kappa)
        if s_pos > h or s_neg < -h:
            return t + 1
    return int(z.size)
