# Monte Carlo Project 2

Project for pricing an up-and-out call option using Monte Carlo
and comparing estimator variants: crude, stratified, antithetic,
control variate. It also includes scripts for generating plots
and notebooks with experiments.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

Estimator value vs. number of paths R:

```bash
python plots/plot_estimator_vs_R.py --method crude --output plots/estimator_vs_R.html
```

Comparison of all estimators (saved as a Plotly .pkl object):

```bash
python scripts/generate_estimators_plot.py --output plots/estimators_vs_R.pkl
```

Visualization of sample GBM trajectories (opens a plot window):

```bash
python scripts/trajectories_plot.py
```

## Project structure

- `methods/` - Monte Carlo estimator implementations
- `models/` - process simulation (GBM)
- `payoffs/` - payoff functions (barrier option)
- `parameters.py` - default parameters
- `plots/` - generated plots and scripts to create them
- `notebooks/` - notebooks (split into `plots/`, `parameters/`, `general/`)

## Notebooks

Notebooks are grouped in `notebooks/` by topic:
plots (`notebooks/plots/`), parameters (`notebooks/parameters/`),
general sketches (`notebooks/general/`).

## Documentation

The assignment and partial results are described in `MC_2025_P2.pdf`.
