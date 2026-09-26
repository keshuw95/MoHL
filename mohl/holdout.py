"""Held-out split of the observed cells.

MoHL learns its weights by predicting observed cells that it hides from
itself. The hidden cells mimic the mask:

  * sensors -- if some sensors never report, the same share of reporting
               sensors is hidden completely;
  * events  -- on the other sensors, gaps are removed whose lengths are
               resampled from the mask's own gap lengths, spread over the
               estimated number of co-missing neighbors.

Events starting in even day blocks form Omega_learn (the training targets)
and those in odd day blocks form Omega_sel, so the two are disjoint in time.
Omega_fit is the rest of the observed cells.
"""
from __future__ import annotations

import numpy as np

from mohl import cadence

__all__ = ["temporal_gap", "holdout_split"]


def temporal_gap(M, cap=None):
    """Steps from each cell to the nearest observed cell of the same sensor
    (0 if observed), capped at one day by default."""
    cap = cap or cadence.steps_per_day()
    N, T = M.shape
    obs = M > 0
    idx = np.arange(T)[None, :]
    last = np.maximum.accumulate(np.where(obs, idx, -10 ** 9), axis=1)
    nxt = np.minimum.accumulate(np.where(obs, idx, 10 ** 9)[:, ::-1], axis=1)[:, ::-1]
    gap = np.minimum(idx - last, nxt - idx)
    return np.minimum(gap, cap).astype(float)


def gap_lengths(M):
    """Lengths of the missing stretches of sensors that report at least once."""
    out = []
    for row in (M == 0):
        if not row.any() or row.all():
            continue
        idx = np.flatnonzero(np.diff(np.concatenate(([0], row.view(np.int8), [0]))))
        out.extend((idx[1::2] - idx[0::2]).tolist())
    return np.asarray(out, dtype=int)


def cluster_size(M, D, n_probe=4, max_cluster=8):
    """Number of sensors that go missing together: 1 + the mean number of
    co-missing nearest neighbors, if it clearly exceeds independence."""
    miss = M == 0
    N = M.shape[0]
    if D is None or N <= n_probe:
        return 1
    Dw = np.array(D, dtype=float).copy()
    np.fill_diagonal(Dw, np.inf)
    nn = np.argsort(Dw, axis=1)[:, :n_probe]
    co_counts, has_miss = np.zeros(N), np.zeros(N)
    for i in range(N):
        mi = miss[i]
        if not mi.any():
            continue
        co_counts[i] = miss[nn[i]][:, mi].sum(axis=0).mean()
        has_miss[i] = 1.0
    if has_miss.sum() == 0:
        return 1
    mean_co = co_counts[has_miss > 0].mean()
    excess = max(mean_co - n_probe * miss.mean(), 0.0)
    return int(np.clip(round(1 + mean_co), 1, max_cluster)) if excess > 0.25 else 1


def holdout_split(M_obs, rho_val, seed, D=None, block_len=None):
    """(M_fit, M_learn, M_sel, info); 2 * rho_val of the observed cells
    (at most 0.6) are hidden and split between M_learn and M_sel."""
    block_len = block_len or cadence.steps_per_day()      # one day
    N, T = M_obs.shape
    rng = np.random.default_rng(50_000 + seed)
    obs = M_obs > 0
    dead = ~obs.any(1)
    live = np.flatnonzero(~dead)
    info = {"silent_sensors": int(dead.sum())}
    learn = np.zeros((N, T), bool)
    sel = np.zeros((N, T), bool)

    # sensor level: hide whole reporting sensors in the share of silent ones
    val_rows = np.array([], dtype=int)
    if dead.any():
        frac = float(np.clip(dead.mean(), 0.10, 0.30))
        n_val = int(np.clip(2 * round(frac * len(live) / 2), 2, len(live) - 2))
        val_rows = rng.choice(live, size=n_val, replace=False)
        learn[val_rows[0::2]] = obs[val_rows[0::2]]
        sel[val_rows[1::2]] = obs[val_rows[1::2]]
    info["held_out_sensors"] = int(len(val_rows))

    # event level: gaps with resampled lengths on the remaining sensors
    rows = np.setdiff1d(live, val_rows)
    lengths = gap_lengths(M_obs)
    n_ev = 0
    if len(lengths) and len(rows):
        k = 1 if dead.any() else cluster_size(M_obs, D)
        nn = None
        if D is not None and k > 1:
            Dw = np.array(D, float).copy()
            np.fill_diagonal(Dw, -1.0)
            nn = np.argsort(Dw, axis=1)
        target = min(2.0 * rho_val, 0.6) * obs[rows].sum()
        taken = np.zeros((N, T), bool)
        ev_id = np.full((N, T), -1, dtype=np.int64)
        ev_start = []
        count = 0
        while count < target and n_ev < 500_000:
            i = int(rng.choice(rows))
            ln = int(rng.choice(lengths))
            s = int(rng.integers(0, max(1, T - ln + 1)))
            mem = nn[i, :k] if nn is not None else np.array([i])
            mem = mem[np.isin(mem, rows)]
            seg = slice(s, min(s + ln, T))
            got = 0
            for r in mem:
                new = obs[r, seg] & ~taken[r, seg]
                if new.any():
                    taken[r, seg] |= new
                    ev_id[r, seg][new] = n_ev
                    got += int(new.sum())
            if got:
                count += got
                n_ev += 1
                ev_start.append(s)
        to_learn = (np.asarray(ev_start, int) // block_len) % 2 == 0
        if n_ev == 0:
            to_learn = np.ones(1, bool)
        in_learn = taken & to_learn[np.maximum(ev_id, 0)]
        learn |= in_learn
        sel |= taken & ~in_learn
        info.update(cluster=k, median_gap=float(np.median(lengths)))
    info["events"] = n_ev
    M_learn, M_sel = learn.astype(float), sel.astype(float)
    M_fit = M_obs * (1.0 - M_learn) * (1.0 - M_sel)
    info.update(n_fit=int(M_fit.sum()), n_learn=int(M_learn.sum()), n_sel=int(M_sel.sum()))
    return M_fit, M_learn, M_sel, info
