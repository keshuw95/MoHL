"""Run MoHL under the paper's protocol and report MAE on the hidden cells.

    python run_mohl.py --dataset pems-bay --regimes cell cluster5 block288
    python run_mohl.py --dataset pems03 --regimes block288 --ablation

Protocol: the 100 highest-degree sensors, the densest seven-day window
(T = 2016), 50% injected missingness. One JSON per (dataset, regime, seed)
is written to --outdir.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from mohl import MEMBERS, REGIMES, impute, load_dataset, make_masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["pems-bay", "pems03", "pems04"])
    ap.add_argument("--regimes", nargs="+", default=["cell", "cluster5", "block288"],
                    help=f"any of {REGIMES}")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--n-sensors", type=int, default=100)
    ap.add_argument("--T", type=int, default=2016)
    ap.add_argument("--rate", type=float, default=0.5)
    ap.add_argument("--steps", type=int, default=100, help="Adam steps of the first and last stage")
    ap.add_argument("--chain-steps", type=int, default=50, help="Adam steps of the middle stages")
    ap.add_argument("--ablation", action="store_true", help="also fit MoHL-Base and MoHL-Free")
    ap.add_argument("--data-root", default=None, help="default: ./data or $MOHL_DATA")
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--save-pred", action="store_true", help="also save the imputations (.npz)")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    b = load_dataset(args.dataset, n_sensors=args.n_sensors, T=args.T, root=args.data_root)
    print(f"{args.dataset}: {b.Y.shape[0]} sensors x {b.Y.shape[1]} steps ({b.unit})", flush=True)
    for regime in args.regimes:
        for seed in args.seeds:
            t0 = time.time()
            M_obs, M_eval = make_masks(b, regime, args.rate, seed)
            ev = M_eval > 0
            preds, info = impute(b.Y, M_obs, b.A, b.D, seed=seed, n_steps=args.steps,
                                 chain_steps=args.chain_steps, ablation=args.ablation,
                                 verbose=args.verbose)
            mae = {k: float(np.abs(X - b.Y)[ev].mean()) for k, X in preds.items()}
            print(f"{args.dataset}/{regime} seed {seed}: "
                  + "  ".join(f"{k} {v:.4f}" for k, v in mae.items())
                  + f"  [{time.time() - t0:.0f}s]", flush=True)
            stem = os.path.join(args.outdir, f"{args.dataset}__{regime}__s{seed}")
            json.dump({"dataset": args.dataset, "regime": regime, "seed": seed, "unit": b.unit,
                       "mae": mae, "n_eval": int(ev.sum()), "config": vars(args),
                       "n_groups": info["n_groups"], "params": info["params"],
                       "split": info["split"], "seconds": time.time() - t0},
                      open(stem + ".json", "w"), indent=1)
            if args.save_pred:
                names = [k for k in MEMBERS if k in preds]
                np.savez_compressed(stem + ".npz", Y=b.Y.astype(np.float32),
                                    M_obs=M_obs.astype(np.uint8), M_eval=M_eval.astype(np.uint8),
                                    names=np.array(names),
                                    preds=np.stack([preds[k] for k in names]).astype(np.float32))


if __name__ == "__main__":
    main()
