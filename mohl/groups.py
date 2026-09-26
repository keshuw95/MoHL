"""Multiscale groups discovered from the observed cells.

Every step is a function of the observed readings and the mask alone:

  1. daily profile   P_i(tod): per-sensor mean of the observed cells in each
                     time-of-day bin, circularly smoothed.
  2. slow anomaly    S: masked one-hour moving average of (Yz - P).
  3. affinity        pairwise-complete correlation of S; pairs with fewer
                     than `min_common` jointly observed steps are set to 0.
  4. groups          for each order s, every sensor together with its s - 1
                     strongest neighbors. A sensor's order-s group lies inside
                     its order-(s+1) group, so the orders are multiscale.

Each order s contributes one component to the mixture: the adjacency whose
Laplacian is  sum_e (I_e - 11^T/|e|)  over the groups e of order s, i.e. one
weight shared by all pairs of a group.
"""
from __future__ import annotations

import numpy as np

from mohl import cadence

__all__ = ["daily_profile", "slow_anomaly", "anomaly_correlation", "neighbor_groups",
           "group_adjacency", "unit_radius"]


def daily_profile(Yz, M, n_per_day=None, bin_len=None):
    """(N, T) per-sensor daily profile from observed cells only (0 for a
    sensor with no observation)."""
    n_per_day = n_per_day or cadence.steps_per_day()
    bin_len = bin_len or cadence.profile_bin()
    N, T = Yz.shape
    nb = n_per_day // bin_len
    b = (np.arange(T) % n_per_day) // bin_len
    obs = (M > 0).astype(float)
    num = np.zeros((N, nb)); cnt = np.zeros((N, nb))
    for k in range(nb):
        tt = b == k
        num[:, k] = (Yz[:, tt] * obs[:, tt]).sum(1)
        cnt[:, k] = obs[:, tt].sum(1)
    prof = np.where(cnt > 0, num / np.maximum(cnt, 1), np.nan)
    # fill empty bins of a sensor from its own circular neighbours
    for i in range(N):
        r = prof[i]
        if np.isnan(r).all():
            continue
        if np.isnan(r).any():
            good = np.flatnonzero(~np.isnan(r))
            ext = np.concatenate([good - nb, good, good + nb])
            r[:] = np.interp(np.arange(nb), ext, np.tile(r[good], 3))
    prof[np.isnan(prof).all(1)] = 0.0
    # circular smoothing across bins
    ker = np.array([0.25, 0.5, 0.25])
    pad = np.concatenate([prof[:, -1:], prof, prof[:, :1]], axis=1)
    prof = np.stack([np.convolve(r, ker, mode="valid") for r in pad])
    # every timestep by circular linear interpolation between bin centres
    centres = (np.arange(nb) + 0.5) * bin_len - 0.5
    ext = np.concatenate([centres - n_per_day, centres, centres + n_per_day])
    tod = np.arange(n_per_day)
    day = np.stack([np.interp(tod, ext, np.tile(r, 3)) for r in prof])
    return day[:, np.arange(T) % n_per_day]


def slow_anomaly(Yz, M, P, half_width=None, min_count=None):
    """Masked moving average of the anomaly (one hour wide). Returns (S, M_S)."""
    half_width = half_width or cadence.hour_half_width()
    min_count = min_count or min(3, 2 * half_width + 1)
    obs = (M > 0).astype(float)
    ker = np.ones(2 * half_width + 1)
    num = np.stack([np.convolve(r, ker, mode="same") for r in (Yz - P) * obs])
    cnt = np.stack([np.convolve(r, ker, mode="same") for r in obs])
    ok = cnt >= min_count
    return np.where(ok, num / np.maximum(cnt, 1), 0.0), ok.astype(float)


def anomaly_correlation(S, M_S, min_common=100):
    """Pairwise-complete correlation of the slow anomaly; 0 where unsupported."""
    m = M_S
    n = m @ m.T
    nn = np.maximum(n, 1)
    sx = (S * m) @ m.T                      # sum of x over the common support
    sxx = (S * S * m) @ m.T
    sxy = (S * m) @ (S * m).T
    cov = sxy / nn - (sx / nn) * (sx.T / nn)
    vx = sxx / nn - (sx / nn) ** 2
    C = cov / np.sqrt(np.maximum(vx * vx.T, 1e-12))
    C[n < min_common] = 0.0
    np.fill_diagonal(C, 0.0)
    return np.clip(np.nan_to_num(C), -1.0, 1.0)


def neighbor_groups(C, s, c_min=0.2):
    """Groups of order s: each sensor with its s - 1 most correlated neighbors
    (kept if the weakest of them has affinity >= c_min); duplicates removed."""
    out, seen = [], set()
    for i in range(C.shape[0]):
        order = np.argsort(-C[i])[:s - 1]
        if len(order) < s - 1 or C[i, order[-1]] < c_min:
            continue
        g = tuple(sorted([i] + order.tolist()))
        if g not in seen:
            seen.add(g)
            out.append(list(g))
    return out


def group_adjacency(N, groups, weights):
    """Adjacency whose graph Laplacian is  sum_e w_e (I_e - 11^T/|e|)."""
    A = np.zeros((N, N))
    for e, w in zip(groups, weights):
        idx = np.asarray(e, dtype=int)
        A[np.ix_(idx, idx)] += w / len(idx)
    np.fill_diagonal(A, 0.0)
    return A


def unit_radius(A):
    """Scale A so that its graph Laplacian has spectral radius 1."""
    L = np.diag(A.sum(1)) - A
    r = float(np.linalg.eigvalsh(L).max())
    return A / r if r > 0 else A
