"""Traffic benchmarks and missingness regimes.

``load_dataset`` reads a benchmark from ``<root>/<name>/``, keeps the n
highest-degree sensors of the largest connected components of the prior graph
and the densest window of T steps. ``make_masks`` injects a missingness regime
on top of the native one. Layout: ``Y[i, t]`` is the reading of sensor i at
step t; ``M[i, t] = 1`` if it is observed.

Expected files (the unzipped archives linked in the README):
    pems-bay/  pems_bay.h5, distances_bay.csv
    pems03/    pems03.npz,  distances.csv, index.txt
    pems04/    pems04.npz,  distance.csv
The distance matrix is built from the csv on first use and cached as .npy.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from mohl import cadence

DATA_ROOT = os.environ.get(
    "MOHL_DATA", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data"))

META = {
    "pems-bay": dict(unit="mph", h5="pems_bay.h5", csv="distances_bay.csv", dist="pems_bay_dist.npy"),
    "pems03": dict(unit="veh/5min", npz="pems03.npz", csv="distances.csv", ids="index.txt",
                   dist="distance_matrix.npy"),
    "pems04": dict(unit="veh/5min", npz="pems04.npz", csv="distance.csv", dist="distance_matrix.npy"),
}

REGIMES = ["cell", "cluster5", "block6", "block72", "block288", "mixed"]


@dataclass
class DatasetBundle:
    name: str
    Y: np.ndarray      # (N, T) readings, 0 where missing
    M0: np.ndarray     # (N, T) native observation mask (1 = observed)
    A: np.ndarray      # (N, N) prior-graph adjacency
    D: np.ndarray      # (N, N) road distances (shortest paths)
    steps_per_day: int = 288
    unit: str = "mph"


# --------------------------------------------------------------------------- #
# Raw files                                                                   #
# --------------------------------------------------------------------------- #

def _distance_matrix(folder, meta, sensor_ids):
    """Directed road distances from the published csv (inf = no edge)."""
    path = os.path.join(folder, meta["dist"])
    if os.path.exists(path):
        return np.load(path)
    import pandas as pd
    table = pd.read_csv(os.path.join(folder, meta["csv"]))
    pos = {int(s): i for i, s in enumerate(sensor_ids)}
    dist = np.full((len(sensor_ids),) * 2, np.inf, dtype=np.float32)
    for a, b, c in table.values[:, :3]:
        if int(a) in pos and int(b) in pos:
            dist[pos[int(a)], pos[int(b)]] = c
    np.save(path, dist)
    return dist


def load_raw(name, root=None):
    """(Y[N, T], M0[N, T] bool, dist[N, N]) with native missingness kept."""
    meta = META[name]
    folder = os.path.join(root or DATA_ROOT, name)
    if "h5" in meta:
        import pandas as pd
        df = pd.read_hdf(os.path.join(folder, meta["h5"]))
        idx = pd.date_range(df.index[0], df.index[-1], freq="5min")
        df = df.reindex(index=idx)
        Y = df.values.astype(float)
        M0 = ~np.isnan(Y) & (Y != 0)          # 0 is the missing sentinel
        Y = np.nan_to_num(Y)
        ids = [int(c) for c in df.columns]
    else:
        fp = np.load(os.path.join(folder, meta["npz"]))
        data = fp["data"]; fp.close()
        Y = data[..., 0].astype(float)         # flow channel
        M0 = Y != 0
        ids = (np.loadtxt(os.path.join(folder, meta["ids"]), dtype=int) if "ids" in meta
               else np.arange(Y.shape[1]))
    dist = _distance_matrix(folder, meta, ids)
    return Y.T, M0.T, np.asarray(dist, float)


def distance_adjacency(dist):
    """Gaussian-kernel adjacency (theta = std of finite distances, threshold
    0.1), symmetrised, as in DCRNN."""
    D = np.array(dist, float)
    finite = np.isfinite(D) & (D > 0)
    A = np.zeros_like(D)
    theta = np.std(D[finite])
    A[finite] = np.exp(-(D[finite] / theta) ** 2)
    A[A < 0.1] = 0.0
    np.fill_diagonal(A, 0.0)
    return np.maximum(A, A.T)


def shortest_path_distances(dist):
    """All-pairs shortest paths on a sparse distance graph (inf = no edge)."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import shortest_path
    D = np.array(dist, float)
    finite = np.isfinite(D)
    np.fill_diagonal(finite, False)
    W = np.where(finite, D, 0.0)
    SP = shortest_path(csr_matrix(W), method="D", directed=False)
    np.fill_diagonal(SP, 0.0)
    return SP


def select_subnetwork(A, n):
    """The n highest-degree sensors of the largest connected components."""
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    N = A.shape[0]
    if n is None or n >= N:
        return np.arange(N)
    _, lab = connected_components(csr_matrix(A > 0), directed=False)
    # largest components first until they hold n sensors (sparse road graphs
    # such as PEMS03 consist of several chains)
    sizes = np.bincount(lab)
    keep, tot = [], 0
    for c in np.argsort(-sizes, kind="stable"):
        keep.append(c); tot += sizes[c]
        if tot >= n:
            break
    cand = np.flatnonzero(np.isin(lab, keep))
    cand = cand[np.argsort(-A.sum(1)[cand], kind="stable")]
    return np.sort(cand[:n])


def load_dataset(name, n_sensors=100, T=2016, root=None):
    """Sub-network of n_sensors and the densest window of T steps."""
    Y, M0, dist = load_raw(name, root)
    cadence.set_steps_per_day(288)
    A = distance_adjacency(dist)
    D = shortest_path_distances(dist) if np.isfinite(dist).mean() < 0.5 else np.array(dist, float)
    D = np.where(np.isfinite(D), D, np.nanmax(D[np.isfinite(D)]) * 10)
    idx = select_subnetwork(A, n_sensors)
    Y, M0 = Y[idx], M0[idx]
    A, D = A[np.ix_(idx, idx)], D[np.ix_(idx, idx)]
    if T is not None and T < Y.shape[1]:
        obs_t = M0.mean(0)
        best, start = -np.inf, 0
        for s in range(0, Y.shape[1] - T + 1, T // 8 + 1):
            frac = obs_t[s:s + T].mean()
            if frac > best:
                best, start = frac, s
        Y, M0 = Y[:, start:start + T], M0[:, start:start + T]
    Y = Y * M0
    return DatasetBundle(name=name, Y=Y.astype(float), M0=M0.astype(float), A=A, D=D,
                         steps_per_day=288, unit=META[name]["unit"])


# --------------------------------------------------------------------------- #
# Missingness regimes                                                         #
# --------------------------------------------------------------------------- #

def inject_cell(M0, rate, seed):
    """Scattered dropout: each observed cell is hidden with probability rate."""
    rng = np.random.default_rng(seed)
    eligible = M0 > 0
    hold = eligible & (rng.random(M0.shape) < rate)
    return (eligible & ~hold).astype(float), hold.astype(float)


def inject_block(M0, rate, seed, block_len=6):
    """Contiguous gaps of block_len steps, drawn independently per sensor."""
    rng = np.random.default_rng(seed)
    N, T = M0.shape
    hold = np.zeros_like(M0, dtype=bool)
    for i in range(N):
        eligible_t = np.where(M0[i] > 0)[0]
        if len(eligible_t) == 0:
            continue
        n_blocks = int(np.ceil(rate * len(eligible_t) / block_len))
        starts = rng.choice(np.arange(max(1, T - block_len + 1)), size=n_blocks, replace=False)
        for s in starts:
            hold[i, s:s + block_len] = True
    hold &= (M0 > 0)
    return ((M0 > 0) & ~hold).astype(float), hold.astype(float)


def inject_cluster(M0, rate, seed, *, D, cluster_size=5, block_len=6, max_events=200_000):
    """Spatially clustered gaps: each event hides a sensor and its
    cluster_size - 1 nearest neighbours for block_len steps."""
    rng = np.random.default_rng(seed)
    N, T = M0.shape
    eligible = M0 > 0
    target = rate * eligible.sum()
    k = int(min(max(cluster_size, 1), N))
    Dw = np.array(D, dtype=float).copy()
    np.fill_diagonal(Dw, -1.0)                 # self sorts first
    nn = np.argsort(Dw, axis=1)
    hold = np.zeros((N, T), dtype=bool)
    n_events = 0
    while hold[eligible].sum() < target and n_events < max_events:
        i = int(rng.integers(N))
        s = int(rng.integers(0, max(1, T - block_len + 1)))
        hold[np.ix_(nn[i, :k], range(s, min(s + block_len, T)))] = True
        n_events += 1
    hold &= eligible
    return (eligible & ~hold).astype(float), hold.astype(float)


def inject_silent_sensors(M0, rate, seed):
    """A fraction of sensors that never report."""
    rng = np.random.default_rng(seed)
    N, T = M0.shape
    hold_idx = rng.choice(N, size=max(1, int(np.round(rate * N))), replace=False)
    M_obs, M_eval = M0.copy(), np.zeros_like(M0)
    for i in hold_idx:
        M_eval[i] = (M0[i] > 0).astype(float)
        M_obs[i] = 0.0
    return M_obs, M_eval


def make_masks(b, regime, rate=0.5, seed=0):
    """(M_obs, M_eval) for a regime: 'cell', 'block<L>' (L steps),
    'cluster<k>' (k sensors x 30 min) or 'mixed'."""
    if regime == "cell":
        return inject_cell(b.M0, rate, seed)
    if regime.startswith("block"):
        return inject_block(b.M0, rate, seed, block_len=int(regime[5:]))
    if regime.startswith("cluster"):
        return inject_cluster(b.M0, rate, seed, D=b.D, cluster_size=int(regime[7:]), block_len=6)
    if regime == "mixed":
        # 15% silent sensors, 6-hour gaps on a quarter of what remains,
        # scattered dropout on a quarter of the rest
        M1, _ = inject_silent_sensors(b.M0, 0.15, seed)
        M2, _ = inject_block(M1, 0.25, seed + 101, block_len=cadence.steps(6))
        M3, _ = inject_cell(M2, 0.25, seed + 202)
        return M3, ((b.M0 > 0) & (M3 == 0)).astype(float)
    raise ValueError(f"unknown regime {regime!r}")
