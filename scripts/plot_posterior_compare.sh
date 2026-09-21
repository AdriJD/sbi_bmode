#!/bin/bash

source /u/bing/sbi_bmode/.venv/bin/activate

python /u/bing/sbi_bmode/scripts/plot_posterior_compare.py \
    --posteriors \
        /ptmp/bing/2026_sbi_bmode/run91hf_optuna_a/trial_0138/posterior.pkl \
        /ptmp/bing/2026_sbi_bmode/run90_optuna_b/trial_0032/posterior.pkl \
        /ptmp/bing/2026_sbi_bmode/run89_optuna_g/trial_0133/posterior.pkl \
    --labels \
        'Train: MultiFid ($N_{\mathrm{transfer}}=20480,\ N_{\mathrm{obsmat}}=2048$), Test: ObsMat' \
        'Train: ObsMat $N=20480$, Test: ObsMat' \
        'Train: Transfer $N=20480$, Test: ObsMat' \
    --configs \
        /ptmp/bing/2026_sbi_bmode/run91hf_a/config.yaml \
        /ptmp/bing/2026_sbi_bmode/run90t_b/config.yaml \
        /ptmp/bing/2026_sbi_bmode/run89t_obsmat_g/config.yaml \
    --test-data \
        /ptmp/bing/2026_sbi_bmode/run89t_obsmat_g/data_draws_test.npy \
        /ptmp/bing/2026_sbi_bmode/run90t_b/data_draws_test.npy \
        /ptmp/bing/2026_sbi_bmode/run89t_obsmat_g/data_draws_test.npy \
    --test-params \
        /ptmp/bing/2026_sbi_bmode/run89t_obsmat_g/param_draws_test.npy \
        /ptmp/bing/2026_sbi_bmode/run90t_b/param_draws_test.npy \
        /ptmp/bing/2026_sbi_bmode/run89t_obsmat_g/param_draws_test.npy \
    --test-idx-range 0 20 \
    --nsamp 10000 \
    --seed 20 \
    --output \
        /ptmp/bing/2026_sbi_bmode/posterior_comparison/test_new.png