# RIOVSPT-infty: code and data for the numerical experiments

Code and raw results accompanying the paper

> Q. Zhang, Y. Wang, X. Li, J. Jia, X. Guan, P. M. Pardalos,
> *The Restricted Inverse Optimal Value Problem on Shortest Path under the
> l_infinity Norm on Trees* (submitted to Journal of Global Optimization).

The repository contains Python implementations of the paper's Algorithms 1–3,
the epigraph LP baseline (SciPy/HiGHS and Gurobi), the deterministic instance
generators, all raw per-run results, and the scripts that produce the tables
and figures in the experiments section.

## Files

| File | Purpose |
|---|---|
| `algorithms.py` | Algorithm 1 (parametric greedy subproblem), Algorithm 2 (three-case method), Algorithm 3 (uniform threshold method) |
| `study_instances.py` | Tree families, deterministic random generators, 13 named boundary cases |
| `study_runner.py` | LP baseline (HiGHS, Gurobi), solution validation, correctness / performance / sensitivity / large-scale runs |
| `experiment_spit.py` | Validation of Algorithm 1 and the value-function counterexample |
| `experiment_branches.py` | Branch-wise runtime comparison of Algorithms 2 and 3 |
| `summarize_results.py` | Reconciles all CSV files and writes `results/full_summary.json` |
| `make_figures.py`, `make_tree_families.py` | Figures |
| `smoke_test.py` | Quick check on the 13 named cases (a few seconds) |
| `results/` | Raw results used in the paper (per-run CSV, seeds, summaries, environment) |
| `figures/` | Figures used in the paper (PDF and PNG) |

## Requirements

Python 3.12, NumPy, SciPy and Matplotlib (`pip install -r requirements.txt`).
The Gurobi baseline additionally needs `gurobipy` and a valid Gurobi licence;
the algorithms themselves and the HiGHS baseline do not.

## Reproducing the results

Run the timing groups serially, not in parallel:

```
python smoke_test.py
python study_runner.py --in-process --write-plan --run-correctness --run-performance --run-sensitivity --run-large --out results
python experiment_spit.py --out results
python experiment_branches.py --out results
python summarize_results.py --results results
python make_figures.py --results results --out figures
python make_tree_families.py --out figures
```

Every instance is reconstructed from its family, size, parameters and the
`actual_seed` column in the CSV files; `results/study_plan.json` lists all seeds.

## Timing protocol

Each method gets one warm-up call and three measured serial calls per instance;
the reported time is the median of the three. Process start-up, file I/O and
the independent solution check are excluded. LP times include model
construction. HiGHS and Gurobi are restricted to one thread.

The recorded results were produced with CPython 3.12.6, NumPy 2.4.3,
SciPy 1.18.1 and Gurobi 12.0.1 on an Intel Core i9-14900HX under Windows 11
(see `results/environment.txt` and `results/run_log.txt`).

## Licence

MIT, see `LICENSE`. If you use this code, please cite the paper above.
