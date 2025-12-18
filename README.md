# sbi_bmode

Code supporting arxiv:XXXX.XXXXX. `sbi_bmode` is a python library that combines Needlet-ILC with simulation-based inference to estimate the tensor-to-scalar ratio from CMB B-mode polarization data in the presence of complicated Galactic foregrounds.

### Dependencies

- [numpy](https://github.com/numpy/)
- [scipy](https://github.com/scipy/scipy)
- [matplotlib](https://github.com/matplotlib/matplotlib)
- [mpi4py](https://github.com/mpi4py/mpi4py)
- [pytorch](https://github.com/pytorch/pytorch)
- [jax](https://github.com/jax-ml/jax)
- [sbi](https://github.com/sbi-dev/sbi)
- [pyilc](https://github.com/jcolinhill/pyilc)
- [blackjax](https://github.com/blackjax-devs/blackjax)
- [tarp](https://github.com/Ciela-Institute/tarp)
- [healpy](https://github.com/healpy/healpy)
- [optweight](https://github.com/AdriJD/optweight)
- [pixell](https://github.com/simonsobs/pixell)

### Installation
```
pip install .
```
Consider adding the `-e` flag (`pip install -e .`) to enable automatic 
updating of code changes when developing.

### Usage

The main script is [run_sbi_basic.py](scripts/run_sbi_basic.py). See [runs/README](scripts/runs/README) for examples of slurm scripts used to run the scripts. The [scripts/paper](scripts/paper) directory contains scripts to produce the plots in the paper.
