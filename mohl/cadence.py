"""Sampling cadence of the loaded dataset.

Every step count that encodes physical time (the daily period, the profile
bin, the one-hour moving average, the day blocks of the held-out split and
the cap of the temporal gap) derives from ``steps_per_day``, which
``mohl.data.load_dataset`` sets. The default is 288 steps per day (5 min).
"""
from __future__ import annotations

_STEPS_PER_DAY = 288


def set_steps_per_day(n: int) -> None:
    global _STEPS_PER_DAY
    _STEPS_PER_DAY = int(n)


def steps_per_day() -> int:
    return _STEPS_PER_DAY


def steps(hours: float) -> int:
    """Number of steps in `hours` at the current cadence (at least 1)."""
    return max(1, int(round(hours * _STEPS_PER_DAY / 24)))


def profile_bin() -> int:
    """Profile bin: 30 min at 5-min cadence (6 steps); one step when coarser."""
    return max(1, _STEPS_PER_DAY // 48)


def hour_half_width() -> int:
    """Half-width of the one-hour moving average (6 at 5 min; >= 1)."""
    return max(1, _STEPS_PER_DAY // 48)
