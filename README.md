# I-ARJ8S: An Inertial, Non-Monotone Globalization of the Eighth-Order ARJ8S Method

Reference implementation and reproducibility code for the manuscript
*"An Inertial, Non-Monotone Globalization of the Eighth-Order ARJ8S Method
with an Inexact Krylov Extension for Large-Scale Nonlinear Systems,"*
submitted to *Numerical Linear Algebra with Applications* (Wiley).

**Authors:** Pairat Janngam, Chalermwut Comemuang
(Faculty of Education, Mathematics Program, Buriram Rajabhat University, Thailand)
**Corresponding author:** Chalermwut Comemuang — chalermwut.cm@bru.ac.th

I-ARJ8S globalizes ARJ8S (Cordero, Rojas-Hiciano & Torregrosa, *Appl. Numer.
Math.* 229 (2026) 209-236), an eighth-order Super-Ostrowski iterative scheme
for nonlinear systems $F(x)=0$, by replacing its monotone Armijo
globalization with (i) a safeguarded FISTA-style inertial extrapolation
step, (ii) a DF-SANE-style non-monotone line search applied independently at
each of the four Super-Ostrowski stages, and (iii) an optional inexact
Newton-GMRES/ILU solve for large sparse Jacobians.

This repository contains the full, independent implementation used to
generate every numerical result, table, and figure in the manuscript —
including the comparator methods, the basins-of-attraction and efficiency
plots, the high-precision (800/1000-digit) convergence-order verification,
and the large-scale 2D/3D Bratu and Burgers experiments.

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.9+. `pyamg` is only needed for the AMG-preconditioner
ablation (`krylov_ablation*.py`); everything else runs with just `numpy`,
`scipy`, `matplotlib`, and `mpmath`.

Several scaling/stress scripts import Python's `resource` module for
peak-memory reporting, which is POSIX-only (Linux/macOS); on Windows, either
run them under WSL or comment out the memory-reporting calls.

## Repository layout

**Core method implementations**
- `base_methods.py` — classical Newton, ARJ8S, ARJ8S-Armijo, and the ACOC
  (apparent convergence order) estimator used throughout.
- `iarj8s_inertial.py` — I-ARJ8S itself: inertial extrapolation, non-monotone
  line search, and the optional inexact-GMRES stage solves.
- `comparator_methods.py` — CTT8S, ZO8S, EH9S, as implemented directly from
  the cited comparator papers.
- `nlm8_method.py` — NLM8 (Sharma & Arora, via Zhanlav, Mijiddorj & Khuder).
- `newton_gmres.py` — plain Newton-GMRES with Eisenstat-Walker forcing terms,
  the inexact-Newton baseline I-ARJ8S-GMRES is compared against.
- `idrs_solver.py` — IDR(s) (Sonneveld & van Gijzen 2008), implemented here
  since it is not available in SciPy.

**Test problems**
- `bratu_test.py`, `bratu3d_test.py` — 2D/3D Bratu problem builders.
- `burgers_test.py`, `burgers_scalability.py` — 1D viscous Burgers equation.

**Order verification (the paper's high-precision checks)**
- `verify_arj8s.py`, `verify_arj8s2.py`, `verify_arj8s_mp.py` — ARJ8S order
  checks (the `_mp` variant uses `mpmath` at 800-digit precision).
- `verify_iarj8s.py`, `verify_iarj8s_mp.py` — the corresponding I-ARJ8S checks.
- `paper_own_tests.py`, `vpa_full_size_tests.py` — reproduction of the base
  ARJ8S paper's own Section 7 benchmark problems at their original
  variable-precision-arithmetic sizes and precision.

**Basins of attraction and efficiency figures**
- `basins_test.py` — basins of attraction for ARJ8S vs. I-ARJ8S.
- `efficiency_plots.py` — classical/computational efficiency index and
  computational performance figures (reproduces and extends the base paper's
  Figs. 1-3).

**Large-scale and scaling experiments**
- `bratu_extreme_stress.py`, `bratu_stress_test.py` — 2D Bratu stress tests
  near the bifurcation threshold, up to the base paper's own extreme-scale
  test (4,194,304 unknowns).
- `bratu3d_scaling.py`, `bratu3d_full_comparison.py` — 3D Bratu scaling and
  full method comparison (direct sparse LU vs. GMRES/ILU).
- `bratu3d_bifurcation.py`, `bratu3d_bifurcation_gaps.py` — 3D
  bifurcation-robustness region mapping.
- `large_scale_test.py`, `newton_gmres_scaling.py`,
  `newton_gmres_instrumented_scaling.py` — Newton-GMRES vs. I-ARJ8S-GMRES
  scaling comparisons.
- `gmres_inner_instrumentation.py` — inner GMRES iteration-count
  instrumentation per outer stage.

**Krylov backend and preconditioner ablations**
- `krylov_ablation.py`, `krylov_ablation_standalone.py` — GMRES vs. BiCGSTAB
  vs. TFQMR vs. IDR(s), and ILU vs. AMG preconditioning.
- `precond_reuse_ablation.py` — single reused ILU preconditioner vs.
  per-stage factorization.
- `bicgstab_amg_repeat_test.py` — cross-machine repeatability check for a
  BiCGSTAB+AMG timing discrepancy noted in the manuscript.

**Orchestration**
- `compute_all_tables.py` — master script that reproduces every table in the
  paper that completes in roughly 10-30 minutes at default settings.

## Reproducing the paper's results

```bash
python compute_all_tables.py
```

For the figures:

```bash
python basins_test.py
python efficiency_plots.py
```

For the large-scale 3D Bratu and extreme-stress experiments (each is
self-contained and documents its own expected runtime in its module
docstring — these are not run by `compute_all_tables.py`):

```bash
python bratu_extreme_stress.py
python bratu3d_full_comparison.py
```

## Status

This code accompanies a manuscript currently under review. The local
eighth-order convergence proof for the exact-solve case is established in
the manuscript (Theorem 1); for the inexact-GMRES case, order preservation
is established under an explicit sufficient condition on the forcing-term
schedule (Theorem 2), which is stronger than what the implemented schedule
in `iarj8s_inertial.py` guarantees — see the manuscript's discussion
section for the precise scope of what is proven versus what is verified
empirically here.

## Citation

A citation file (`CITATION.cff`) is included. Please cite the manuscript
once published; in the meantime, cite this repository directly.

## License

MIT — see `LICENSE`.
