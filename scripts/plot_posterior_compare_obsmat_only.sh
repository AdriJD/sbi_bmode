#!/bin/bash

source /u/bing/sbi_bmode/.venv/bin/activate

python /u/bing/sbi_bmode/scripts/plot_posterior_compare.py \
    --posteriors \
        /ptmp/bing/2026_sbi_bmode/run90_optuna_b/trial_0032/posterior.pkl \
        /ptmp/bing/2026_sbi_bmode/run89_optuna_g/trial_0133/posterior.pkl \
    --labels \
        "Train: ObsMat N=20480, Test: ObsMat" \
        "Train: Transfer N=20480, Test: ObsMat" \
    --configs \
        /ptmp/bing/2026_sbi_bmode/run90t_b/config.yaml \
        /ptmp/bing/2026_sbi_bmode/run89t_obsmat_g/config.yaml \
    --test-data \
        /ptmp/bing/2026_sbi_bmode/run90t_b/data_draws_test.npy \
        /ptmp/bing/2026_sbi_bmode/run89t_obsmat_g/data_draws_test.npy \
    --test-params \
        /ptmp/bing/2026_sbi_bmode/run90t_b/param_draws_test.npy \
        /ptmp/bing/2026_sbi_bmode/run89t_obsmat_g/param_draws_test.npy \
    --test-idx 1 \
    --nsamp 10000 \
    --param-idx 0 2 3 \
    --seed 20 \
    --output \
        /ptmp/bing/2026_sbi_bmode/posterior_comparison_obsmat_only/test_zoom_b.png