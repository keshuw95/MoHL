"""MoHL end to end: from an incomplete matrix to its imputation.

    preds, info = impute(Y, M, A, D)          # preds["MoHL"] is the imputation

The fit proceeds in warm-started stages, each a member of the ablation:

    +profile      daily profile + prior graph
    +pairs        + groups of order 2
    MoHL-Static   + groups of orders 3..6 (one shared weight per order)
    MoHL          + evidence-adaptive gate (full model)

With ablation=True two further members are fitted:

    MoHL-Base     prior graph only, no profile
    MoHL-Free     MoHL-Static's support with one free weight per pair

Groups and the gate context are rebuilt from the mask each member is solved
on: from Omega_fit while learning, from the full observed mask at deployment.
"""
from __future__ import annotations

import numpy as np

from mohl.groups import (anomaly_correlation, daily_profile, group_adjacency, neighbor_groups,
                         slow_anomaly, unit_radius)
from mohl.holdout import holdout_split
from mohl.model import cell_context, learn_free, learn_mohl, predict_free, predict_mohl

ORDERS = (2, 3, 4, 5, 6)
MEMBERS = ["MoHL-Base", "+profile", "+pairs", "MoHL-Static", "MoHL", "MoHL-Free"]


def zscore(Y, M):
    """Per-sensor z-scores from observed cells (global statistics for a
    sensor with fewer than two observations). Returns (Yz * M, mean, std)."""
    obs = M > 0
    gm = float(Y[obs].mean()) if obs.any() else 0.0
    gs = float(max(Y[obs].std(), 1e-6)) if obs.any() else 1.0
    mu = np.full(Y.shape[0], gm)
    sd = np.full(Y.shape[0], gs)
    for i in range(Y.shape[0]):
        m = M[i] > 0
        if m.sum() >= 2:
            mu[i] = Y[i, m].mean()
            sd[i] = max(Y[i, m].std(), 1e-6)
    return ((Y - mu[:, None]) / sd[:, None]) * M, mu, sd


def build_structure(Yz, M, A, A_unit, orders, min_common):
    """Profile, gate context and mixture components on mask M."""
    N = M.shape[0]
    Y = Yz * M
    P = daily_profile(Y, M)
    S, MS = slow_anomaly(Y, M, P)
    Cc = anomaly_correlation(S, MS, min_common=min_common)
    comps, groups = [A_unit], {}
    for s in orders:
        g = neighbor_groups(Cc, s)
        groups[s] = g
        comps.append(unit_radius(group_adjacency(N, g, np.ones(len(g)))) if g else np.zeros((N, N)))
    return {"P": P, "C": cell_context(M, A, Y), "comps": comps, "groups": groups,
            "union": sum(comps[1:])}


def impute(Y, M, A, D=None, *, seed=0, orders=ORDERS, n_steps=100, chain_steps=50,
           rho_val=0.2, ablation=False, verbose=False):
    """Impute the missing cells of Y.

    Y : (N, T) readings (any value where M = 0);  M : (N, T) 1 = observed;
    A : (N, N) prior-graph adjacency;  D : (N, N) distances, used to shape the
    held-out events like the mask's gaps (optional).
    Returns ({member: (N, T) imputation in the units of Y}, info).
    """
    N, T = Y.shape
    Y = np.where(M > 0, np.asarray(Y, float), 0.0)
    A_unit = unit_radius(A)
    min_common = max(30, min(100, T // 10))
    Yz, mu, sd = zscore(Y, M)
    M_fit, M_learn, M_sel, split = holdout_split(M, rho_val, seed, D=D)
    fit = build_structure(Yz, M_fit, A, A_unit, orders, min_common)
    obs = build_structure(Yz, M, A, A_unit, orders, min_common)
    K = 1 + len(orders)
    Z0 = np.zeros((N, T))
    static = dict(gate_S=False, gate_T=False)

    def upto(j):                                  # components 0..j present
        return np.arange(K) <= j

    def learn(use_profile, **kw):
        P = fit["P"] if use_profile else Z0
        return learn_mohl(Yz - P, M_fit, M_learn, fit["comps"], fit["C"], verbose=verbose, **kw)

    def deploy(f, use_profile):
        P = obs["P"] if use_profile else Z0
        return predict_mohl(f, Yz - P, M, obs["comps"], obs["C"]) + P

    fits, Xz = {}, {}
    if ablation:
        fits["MoHL-Base"] = learn(False, free_comp=upto(0), n_steps=n_steps, **static)
        Xz["MoHL-Base"] = deploy(fits["MoHL-Base"], False)
    stages = [("+profile", dict(free_comp=upto(0), n_steps=n_steps, **static)),
              ("+pairs", dict(free_comp=upto(1), n_steps=chain_steps, **static)),
              ("MoHL-Static", dict(n_steps=chain_steps, **static)),
              ("MoHL", dict(n_steps=n_steps))]
    prev = None
    for name, kw in stages:
        if verbose:
            print(f"  {name}", flush=True)
        fits[name] = learn(True, init=prev, **kw)
        Xz[name] = deploy(fits[name], True)
        prev = fits[name]["params"]
    if ablation:
        f = learn_free(Yz - fit["P"], M_fit, M_learn, A_unit, fit["union"],
                       init=fits["+profile"]["params"], n_steps=n_steps)
        Xz["MoHL-Free"] = predict_free(f, Yz - obs["P"], M, A_unit, obs["union"]) + obs["P"]
        fits["MoHL-Free"] = f

    preds = {k: Xz[k] * sd[:, None] + mu[:, None] for k in MEMBERS if k in Xz}
    info = {"split": split,
            "n_groups": {str(s): len(obs["groups"][s]) for s in orders},
            "groups": {str(s): obs["groups"][s] for s in orders},
            "params": {k: _readable(fits[k]["params"], fits[k].get("keep"), orders)
                       for k in fits}}
    return preds, info


def _readable(p, keep, orders):
    """Mixture weights alpha (prior graph and each order), lam_T and mu."""
    names = ["prior"] + [f"order{s}" for s in orders]
    out = {"lam_T": float(np.exp(p["log_lam_T"])), "mu": float(np.exp(p["log_mu"]))}
    if keep is not None:
        out["alpha"] = {names[k]: float(np.exp(a)) for k, a in zip(keep, p["a"])}
        out["gate_slopes_norm"] = float(np.linalg.norm(p["b"]))
    else:
        out["alpha"] = {"prior": float(np.exp(p["a"][0])), "pairs": float(np.exp(p["a"][1]))}
    return out
