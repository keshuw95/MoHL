# MoHL: Mixtures of Hypergraph Laplacians

Code for *Learning Higher-Order Interactions from Incomplete Spatiotemporal Data with Mixtures of Hypergraph Laplacians*.

MoHL fills in missing values in a sensor × time matrix. It runs on a CPU with NumPy and SciPy.

## Install

```bash
pip install -r requirements.txt   # Python >= 3.9
```

## Data

Download PEMS-BAY, PEMS03 and PEMS04 (from [Torch Spatiotemporal](https://github.com/TorchSpatiotemporal/tsl)) into `data/`:

```bash
URL=https://huggingface.co/datasets/TorchSpatiotemporal/ProcessedDatasets/resolve/v1.0.0/traffic
curl -L -o pems_bay.zip $URL/pems_bay.zip && unzip pems_bay.zip -d data/pems-bay
curl -L -o pems03.zip   $URL/pems03.zip   && unzip pems03.zip   -d data/pems03
curl -L -o pems04.zip   $URL/pems04.zip   && unzip pems04.zip   -d data/pems04
```

To keep the data elsewhere, set `MOHL_DATA` or pass `--data-root`.

## Run

```bash
python run_mohl.py --dataset pems-bay --regimes cell cluster5 block288
```

- `--dataset`: `pems-bay`, `pems03` or `pems04`
- `--regimes`: any of `cell`, `cluster5`, `block6`, `block72`, `block288`, `mixed`
- `--seeds`: e.g. `--seeds 0 1 2 3 4`
- `--ablation`: also fit `MoHL-Base` and `MoHL-Free`
- `--save-pred`: also save the imputations as `.npz`

The script prints the MAE on the hidden cells and writes one JSON per run to `results/`.

## Use on your own data

```python
from mohl import impute

# Y: (N, T) readings; M: (N, T) mask, 1 = observed
# A: (N, N) graph adjacency; D: (N, N) sensor distances (optional)
preds, info = impute(Y, M, A, D)
X = preds["MoHL"]   # imputed matrix
```

The code assumes 288 steps per day (5-minute data). For other rates, call `mohl.cadence.set_steps_per_day(n)` first.
