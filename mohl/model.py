"""Mixture of hypergraph Laplacians: estimator, adjoint learning, gate.

Estimator. For observation weights W (the mask), component adjacencies A_k
(k = 0: prior graph; k >= 1: the groups of one order), the imputation X
solves the symmetric positive definite system

    A(X) = W.Y ,    A(Z) = W.Z + sum_k e^{a_k} S_k(Z; H_k) + lam_T Z L_T(g) + mu Z,

with L_T the first-difference temporal Laplacian (scaled by 1/4) and

    S_k(Z; H) = 1/2 [ H.(L_k Z) + (A_k H).Z - A_k (H.Z) ],   L_k = D_k - A_k.

Gate. Edge (i, j) of component k at time t has weight
e^{a_k} A_k[i,j] (h^k_it + h^k_jt) / 2 with h^k_it = exp(c_it . b_k), where
c_it is an observation-only context (FEATURES); the temporal edge (i,t)-(i,t+1)
is gated the same way. b = 0 gives h = 1 and the static mixture
sum_k e^{a_k} L_k exactly (MoHL-Static).

Learning. The parameters (a, b, log lam_T, b_T, log mu) minimise the Huber
loss on held-out cells M_learn of the estimate solved on M_fit, plus
gamma * sum_{k>=1} e^{a_k} and a ridge on the gate slopes. With one adjoint
solve A(Lam) = dloss/dX, every gradient is an inner product:

    Q_k  = 1/2 [ Lam.X.deg_k - Lam.(A_k X) - X.(A_k Lam) + A_k (Lam.X) ]
    dl/da_k = -e^{a_k} sum H_k.Q_k         dl/db_k = -e^{a_k} sum H_k.Q_k c
    R    = C_T (Lam_t - Lam_t+1).(X_t - X_t+1)
    dl/dlog lam_T = -lam_T sum g.R          dl/db_T = -lam_T sum g.R cbar
    dl/dlog mu    = -mu <Lam, X>
"""
from __future__ import annotations

import numpy as np

from mohl import cadence
from mohl.holdout import temporal_gap

__all__ = ["FEATURES", "cell_context", "learn_mohl", "predict_mohl",
           "learn_free", "predict_free"]

C_T = 0.25                      # L_T is the first-difference Laplacian divided by 4
HP0 = dict(lam_S=1.0, lam_T=20.0, mu=0.02)          # initial values
FEATURES = ["gap", "nbr_dark", "tod_s1", "tod_c1", "tod_s2", "tod_c2", "state"]


# --------------------------------------------------------------------------- #
# Gate context                                                                #
# --------------------------------------------------------------------------- #

def cell_context(M, A, Yz, n_per_day=None):
    """(N, T, F) observation-only context with fixed scalings, so the same gate
    applies to the fitting mask and to the deployment mask: the temporal gap,
    the share of unobserved graph neighbors, two daily harmonics and the
    network state (smoothed mean observed anomaly)."""
    n_per_day = n_per_day or cadence.steps_per_day()
    N, T = M.shape
    obs = (M > 0).astype(float)
    gap = np.log1p(temporal_gap(M)) / np.log1p(cadence.steps_per_day())
    Ar = A / np.maximum(A.sum(1, keepdims=True), 1e-12)
    nbr_dark = 1.0 - Ar @ obs
    ph = 2 * np.pi * (np.arange(T) % n_per_day) / n_per_day
    cnt = obs.sum(0)
    s = np.where(cnt > 0, (Yz * obs).sum(0) / np.maximum(cnt, 1), 0.0)
    w = 2 * cadence.hour_half_width()                    # one-hour smoothing
    s = np.convolve(s, np.ones(w) / w, mode="same")
    state = np.clip(s, -3, 3) / 3.0

    def bt(v):
        return np.broadcast_to(v[None, :], (N, T))

    feats = {"gap": gap, "nbr_dark": nbr_dark,
             "tod_s1": bt(np.sin(ph)), "tod_c1": bt(np.cos(ph)),
             "tod_s2": bt(np.sin(2 * ph)), "tod_c2": bt(np.cos(2 * ph)),
             "state": bt(state)}
    return np.stack([feats[n] for n in FEATURES], axis=-1)


# --------------------------------------------------------------------------- #
# Operator and solver                                                         #
# --------------------------------------------------------------------------- #

class GatedOperator:
    """A(Z) = W.Z + sum_k e^{a_k} S_k(Z; H_k) + lam_T Z L_T(g) + mu Z."""

    def __init__(self, W, comps, a, H, lam_T, mu, g=None):
        self.W, self.comps, self.a, self.H = W, comps, np.asarray(a, float), H
        self.lam_T, self.mu, self.g = lam_T, mu, g
        self.deg = [A.sum(1) for A in comps]
        self.AH = [A @ h for A, h in zip(comps, H)]
        T = W.shape[1]
        gg = np.ones((1, T - 1)) if g is None else g
        dT = np.zeros((gg.shape[0], T))
        dT[:, :-1] += gg
        dT[:, 1:] += gg
        diag = W + mu + lam_T * C_T * dT
        for k in range(len(comps)):
            diag = diag + 0.5 * np.exp(self.a[k]) * (H[k] * self.deg[k][:, None] + self.AH[k])
        self.Minv = 1.0 / diag                           # Jacobi preconditioner

    def lt(self, Z):
        d = Z[:, :-1] - Z[:, 1:]
        if self.g is not None:
            d = d * self.g
        out = np.zeros_like(Z)
        out[:, :-1] += d
        out[:, 1:] -= d
        return C_T * out

    def __call__(self, Z):
        out = self.W * Z + self.mu * Z + self.lam_T * self.lt(Z)
        for k, A in enumerate(self.comps):
            h = self.H[k]
            out += 0.5 * np.exp(self.a[k]) * (
                h * (self.deg[k][:, None] * Z - A @ Z) + self.AH[k] * Z - A @ (h * Z))
        return out

    def solve(self, rhs, X0=None, tol=1e-7, max_iter=300):
        """Preconditioned conjugate gradients."""
        X = np.zeros_like(rhs) if X0 is None else X0.copy()
        r = rhs - self(X)
        z = self.Minv * r
        p = z.copy()
        rz = float(np.sum(r * z))
        bnorm = float(np.sqrt(np.sum(rhs * rhs))) or 1.0
        for _ in range(max_iter):
            if np.sqrt(float(np.sum(r * r))) <= tol * bnorm:
                break
            Ap = self(p)
            denom = float(np.sum(p * Ap))
            if denom <= 0:
                break
            al = rz / denom
            X += al * p
            r -= al * Ap
            z = self.Minv * r
            rz_new = float(np.sum(r * z))
            p = z + (rz_new / rz) * p
            rz = rz_new
        return X


def _gate(C, b, clip=6.0):
    return np.exp(np.clip(C @ b, -clip, clip))


def build_operator(params, W, comps, C, gate_T=True):
    """params: dict(a (K,), b (K, F), log_lam_T, b_T (F,), log_mu)."""
    H = [_gate(C, params["b"][k]) for k in range(len(comps))]
    g = _gate(0.5 * (C[:, :-1, :] + C[:, 1:, :]), params["b_T"]) if gate_T else None
    return GatedOperator(W, comps, params["a"], H, float(np.exp(params["log_lam_T"])),
                         float(np.exp(params["log_mu"])), g)


def huber_loss_grad(X, Y, mask, delta=1.0):
    n = float(max(mask.sum(), 1.0))
    d = (X - Y) * (mask > 0)
    ad = np.abs(d)
    loss = float(np.sum(np.where(ad <= delta, 0.5 * d * d, delta * (ad - 0.5 * delta))) / n)
    return loss, np.clip(d, -delta, delta) / n


def _adam(p, gr, m, v2, step, lr):
    for k in p:
        g_ = np.asarray(gr[k], float)
        m[k] = 0.9 * m[k] + 0.1 * g_
        v2[k] = 0.999 * v2[k] + 0.001 * g_ * g_
        mh = m[k] / (1 - 0.9 ** (step + 1))
        vh = v2[k] / (1 - 0.999 ** (step + 1))
        p[k] = np.asarray(p[k], float) - lr * mh / (np.sqrt(vh) + 1e-8)


# --------------------------------------------------------------------------- #
# MoHL: shared weight per order, optional gate                                #
# --------------------------------------------------------------------------- #

def loss_and_grad(params, Yz, M_fit, M_learn, comps, C, *, gate_T=True,
                  l1=1e-3, ridge=1e-3, warm=None, tol=1e-7, max_iter=300):
    """Held-out objective and its gradient w.r.t. every parameter (one adjoint)."""
    op = build_operator(params, M_fit, comps, C, gate_T)
    X = op.solve(M_fit * Yz, X0=None if warm is None else warm[0], tol=tol, max_iter=max_iter)
    loss, G = huber_loss_grad(X, Yz, M_learn)
    Lam = op.solve(G, X0=None if warm is None else warm[1], tol=tol, max_iter=max_iter)
    ga = np.zeros(len(comps))
    gb = np.zeros_like(params["b"])
    LX = Lam * X
    for k, A in enumerate(comps):
        Q = 0.5 * (LX * op.deg[k][:, None] - Lam * (A @ X) - X * (A @ Lam) + A @ LX)
        HQ = op.H[k] * Q
        ea = np.exp(params["a"][k])
        ga[k] = -ea * HQ.sum()
        gb[k] = -ea * np.tensordot(HQ, C, axes=([0, 1], [0, 1]))
    R = C_T * (Lam[:, :-1] - Lam[:, 1:]) * (X[:, :-1] - X[:, 1:])
    if gate_T:
        gR = op.g * R
        Cb = 0.5 * (C[:, :-1, :] + C[:, 1:, :])
        g_bT = -op.lam_T * np.tensordot(gR, Cb, axes=([0, 1], [0, 1]))
    else:
        gR = R
        g_bT = np.zeros_like(params["b_T"])
    g_lT = -op.lam_T * gR.sum()
    g_mu = -op.mu * LX.sum()
    # penalties: gamma on the group weights, ridge on every gate slope
    ea = np.exp(params["a"])
    pen = l1 * ea[1:].sum() + 0.5 * ridge * (np.sum(params["b"] ** 2) + np.sum(params["b_T"] ** 2))
    ga[1:] += l1 * ea[1:]
    gb = gb + ridge * params["b"]
    g_bT = g_bT + ridge * params["b_T"]
    grads = {"a": ga, "b": gb, "log_lam_T": g_lT, "b_T": g_bT, "log_mu": g_mu}
    return loss + pen, loss, grads, (X, Lam)


def learn_mohl(Yz, M_fit, M_learn, comps, C, *, free_comp=None, gate_S=True, gate_T=True,
               init=None, n_steps=100, lr=0.1, l1=1e-3, ridge=1e-3,
               a_new=float(np.log(0.25)), verbose=False):
    """Adam on the mixture (and gate) parameters; returns the iterate with the
    lowest objective.

    free_comp : bool (K,), components present in the member (the others are removed).
    gate_S/T  : whether the spatial / temporal gate slopes may move (False: static).
    init      : parameters of the preceding member to warm-start from; new
                components start at a quarter of the prior-graph weight.
    """
    K, F = len(comps), C.shape[-1]
    free_comp = np.ones(K, bool) if free_comp is None else np.asarray(free_comp)
    keep = [k for k in range(K) if free_comp[k]]
    comps_m = [comps[k] for k in keep]
    Km = len(comps_m)
    a0 = np.log(HP0["lam_S"])
    p = {"a": np.array([a0] + [a0 + a_new] * (Km - 1), dtype=float),
         "b": np.zeros((Km, F)), "log_lam_T": float(np.log(HP0["lam_T"])),
         "b_T": np.zeros(F), "log_mu": float(np.log(HP0["mu"]))}
    if init is not None:
        n0 = len(init["a"])
        p["a"][:n0] = init["a"]
        p["b"][:n0] = init["b"]
        p["a"][n0:] = init["a"][0] + a_new
        p["log_lam_T"], p["log_mu"] = init["log_lam_T"], init["log_mu"]
        p["b_T"] = np.array(init["b_T"], float).copy()
    m = {k: np.zeros_like(np.asarray(v, float)) for k, v in p.items()}
    v2 = {k: np.zeros_like(np.asarray(v, float)) for k, v in p.items()}
    best, warm = None, None
    for step in range(n_steps):
        obj, loss, gr, warm = loss_and_grad(p, Yz, M_fit, M_learn, comps_m, C, gate_T=gate_T,
                                            l1=l1, ridge=ridge, warm=warm, max_iter=150)
        if best is None or obj < best[0]:
            best = (obj, {k: np.array(v, float).copy() for k, v in p.items()}, loss)
        if not gate_S:
            gr["b"] = np.zeros_like(gr["b"])
        if not gate_T:
            gr["b_T"] = np.zeros_like(gr["b_T"])
        _adam(p, gr, m, v2, step, lr)
        p["a"] = np.clip(p["a"], np.log(1e-6), np.log(1e4))
        p["log_lam_T"] = float(np.clip(p["log_lam_T"], np.log(1e-4), np.log(1e4)))
        p["log_mu"] = float(np.clip(p["log_mu"], np.log(1e-5), np.log(1e2)))
        p["b"] = np.clip(p["b"], -8, 8)
        p["b_T"] = np.clip(p["b_T"], -8, 8)
        if verbose and step % 20 == 0:
            print(f"    step {step:3d}  held-out loss {loss:.5f}", flush=True)
    out = best[1]
    out["log_lam_T"], out["log_mu"] = float(out["log_lam_T"]), float(out["log_mu"])
    return {"params": out, "keep": keep, "gate_T": gate_T, "loss": best[2]}


def predict_mohl(fit, Yz, M, comps, C):
    """Solve a learned member on mask M (C must be the context of M)."""
    comps_m = [comps[k] for k in fit["keep"]]
    op = build_operator(fit["params"], M, comps_m, C, fit["gate_T"])
    return op.solve(M * Yz, tol=1e-7, max_iter=600)


# --------------------------------------------------------------------------- #
# MoHL-Free: one free weight per pair on the same support                     #
# --------------------------------------------------------------------------- #
#
# A group shares one weight across all its pairs. The support-matched control
# frees them: one weight per pair of the union of the group components,
# learned by the same adjoint on the same objective. For L(A) = D - A,
#     d<Lam, L(A) X>/dA_ij = m_ii + m_jj - m_ij - m_ji ,  m = Lam X^T.

def _apply_LT(Z):
    d = Z[:, :-1] - Z[:, 1:]
    out = np.zeros_like(Z)
    out[:, :-1] += d
    out[:, 1:] -= d
    return C_T * out


def _solve_static(rhs, W, L, lam_T, mu, X0=None, tol=1e-7, max_iter=300):
    """CG on W.Z + L Z + lam_T Z L_T + mu Z = rhs (Jacobi preconditioner)."""
    T = rhs.shape[1]
    dT = np.zeros(T); dT[:-1] += 1.0; dT[1:] += 1.0
    Minv = 1.0 / (W + mu + lam_T * C_T * dT[None, :] + np.diag(L)[:, None])

    def op(Z):
        return W * Z + mu * Z + lam_T * _apply_LT(Z) + L @ Z

    X = np.zeros_like(rhs) if X0 is None else X0.copy()
    r = rhs - op(X)
    z = Minv * r
    p = z.copy()
    rz = float(np.sum(r * z))
    bnorm = float(np.sqrt(np.sum(rhs * rhs))) or 1.0
    for _ in range(max_iter):
        if np.sqrt(float(np.sum(r * r))) <= tol * bnorm:
            break
        Ap = op(p)
        denom = float(np.sum(p * Ap))
        if denom <= 0:
            break
        al = rz / denom
        X += al * p
        r -= al * Ap
        z = Minv * r
        rz_new = float(np.sum(r * z))
        p = z + (rz_new / rz) * p
        rz = rz_new
    return X


def _softplus(x):
    return np.where(x > 30, x, np.log1p(np.exp(np.minimum(x, 30))))


def free_loss_and_grad(p, Yz, M_fit, M_learn, L_prior, ii, jj, base, *,
                       l1=1e-3, warm=None, tol=1e-7, max_iter=300):
    N, T = Yz.shape
    sp0 = float(_softplus(0.0))
    A_f = np.zeros((N, N))
    A_f[ii, jj] = base * _softplus(p["theta"]) / sp0
    A_f = A_f + A_f.T
    L_f = np.diag(A_f.sum(1)) - A_f
    ea = np.exp(p["a"])
    lT, mu = float(np.exp(p["log_lam_T"])), float(np.exp(p["log_mu"]))
    L = ea[0] * L_prior + ea[1] * L_f
    X = _solve_static(M_fit * Yz, M_fit, L, lT, mu,
                      X0=None if warm is None else warm[0], tol=tol, max_iter=max_iter)
    loss, G = huber_loss_grad(X, Yz, M_learn)
    Lam = _solve_static(G, M_fit, L, lT, mu,
                        X0=None if warm is None else warm[1], tol=tol, max_iter=max_iter)
    pen = l1 * ea[1] * A_f[ii, jj].sum()
    mm = Lam @ X.T
    dg = np.diag(mm)
    g_pair = dg[ii] + dg[jj] - mm[ii, jj] - mm[jj, ii]
    sig = 1.0 / (1.0 + np.exp(-p["theta"]))
    gr = {"a": np.array([-ea[0] * np.sum(Lam * (L_prior @ X)),
                         -ea[1] * np.sum(Lam * (L_f @ X)) + pen]),
          "theta": (-ea[1] * g_pair + l1 * ea[1]) * base * sig / sp0,
          "log_lam_T": -lT * np.sum(Lam * _apply_LT(X)),
          "log_mu": -mu * np.sum(Lam * X)}
    return loss + pen, loss, gr, (X, Lam)


def learn_free(Yz, M_fit, M_learn, A_prior, A_support, *, init=None,
               n_steps=100, lr=0.1, l1=1e-3):
    """One free weight per pair on the support of A_support (static).

    A_free = A_support * softplus(theta) / softplus(0), so step 0 is the shared
    operator exactly and any departure is learned from the data.
    """
    N, T = Yz.shape
    iu = np.triu_indices(N, 1)
    sup = A_support[iu] > 0
    ii, jj = iu[0][sup], iu[1][sup]
    base = A_support[ii, jj]
    L_prior = np.diag(A_prior.sum(1)) - A_prior
    a0 = np.log(HP0["lam_S"])
    p = {"a": np.array([a0, a0 + np.log(0.25)]), "theta": np.zeros(len(base)),
         "log_lam_T": float(np.log(HP0["lam_T"])), "log_mu": float(np.log(HP0["mu"]))}
    if init is not None:
        p["a"][0] = init["a"][0]
        p["a"][1] = init["a"][0] + np.log(0.25)
        p["log_lam_T"], p["log_mu"] = init["log_lam_T"], init["log_mu"]
    m = {k: np.zeros_like(np.asarray(v, float)) for k, v in p.items()}
    v2 = {k: np.zeros_like(np.asarray(v, float)) for k, v in p.items()}
    best, warm = None, None
    for step in range(n_steps):
        obj, loss, gr, warm = free_loss_and_grad(p, Yz, M_fit, M_learn, L_prior, ii, jj, base,
                                                 l1=l1, warm=warm, max_iter=150)
        if best is None or obj < best[0]:
            best = (obj, {k: np.array(v, float).copy() for k, v in p.items()}, loss)
        _adam(p, gr, m, v2, step, lr)
        p["a"] = np.clip(p["a"], np.log(1e-6), np.log(1e4))
        p["theta"] = np.clip(p["theta"], -12, 8)
        p["log_lam_T"] = float(np.clip(p["log_lam_T"], np.log(1e-4), np.log(1e4)))
        p["log_mu"] = float(np.clip(p["log_mu"], np.log(1e-5), np.log(1e2)))
    return {"params": best[1], "pairs": (ii, jj), "base": base, "loss": best[2],
            "n_free": int(len(base))}


def predict_free(fit, Yz, M, A_prior, A_support):
    """Deploy learned pair weights on mask M; pairs are matched by identity
    (a pair absent from the deployment support is dropped, a new one keeps
    its shared weight)."""
    N, T = Yz.shape
    p = fit["params"]
    ii, jj = fit["pairs"]
    mult = np.ones((N, N))
    mult[ii, jj] = _softplus(p["theta"]) / float(_softplus(0.0))
    mult = np.triu(mult, 1) + np.triu(mult, 1).T + np.eye(N)
    A_f = A_support * mult
    ea = np.exp(p["a"])
    L = ea[0] * (np.diag(A_prior.sum(1)) - A_prior) + ea[1] * (np.diag(A_f.sum(1)) - A_f)
    return _solve_static(M * Yz, M, L, float(np.exp(p["log_lam_T"])),
                         float(np.exp(p["log_mu"])), tol=1e-7, max_iter=600)
