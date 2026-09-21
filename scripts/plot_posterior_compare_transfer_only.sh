#!/bin/bash

source /u/bing/sbi_bmode/.venv/bin/activate

python /u/bing/sbi_bmode/scripts/plot_posterior_compare.py \
    --posteriors \
        /ptmp/bing/2026_sbi_bmode/run89_optuna_g/trial_0133/posterior.pkl \
        /ptmp/bing/2026_sbi_bmode/run89_optuna_g/trial_0133/posterior.pkl \
    --labels \
        "Train: Transfer , Test: Transfer" \
        "Train: Transfer, Test: ObsMat" \
    --configs \
        /ptmp/bing/2026_sbi_bmode/run89t_g/config.yaml \
        /ptmp/bing/2026_sbi_bmode/run89t_obsmat_g/config.yaml \
    --test-data \
        /ptmp/bing/2026_sbi_bmode/run89t_g/data_draws_test.npy \
        /ptmp/bing/2026_sbi_bmode/run89t_obsmat_g/data_draws_test.npy \
    --test-params \
        /ptmp/bing/2026_sbi_bmode/run89t_g/param_draws_test.npy \
        /ptmp/bing/2026_sbi_bmode/run89t_obsmat_g/param_draws_test.npy \
    --test-idx 0 \
    --nsamp 10000 \
    --seed 20 \
    --output \
        /ptmp/bing/2026_sbi_bmode/posterior_comparison_transfer_only/test_000.png