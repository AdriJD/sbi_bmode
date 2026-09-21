#!/bin/bash

source /u/bing/sbi_bmode/.venv/bin/activate

python /u/bing/sbi_bmode/scripts/plot_posterior_compare.py \
    --posteriors \
        /ptmp/bing/2026_sbi_bmode/run91hf_optuna_f/trial_0092/posterior.pkl \
        /ptmp/bing/2026_sbi_bmode/run90_optuna_b/trial_0032/posterior.pkl \
        /ptmp/bing/2026_sbi_bmode/run89_optuna_g/trial_0133/posterior.pkl \
    --labels \
        'Train: MultiFid ($N_{\mathrm{transfer}}=20480,\ N_{\mathrm{obsmat}}=1152$), Test: ObsMat' \
        'Train: ObsMat $N=20480$, Test: ObsMat' \
        'Train: Transfer $N=20480$, Test: ObsMat' \
    --configs \
        /ptmp/bing/2026_sbi_bmode/run91hf_f/config.yaml \
        /ptmp/bing/2026_sbi_bmode/run91hf_f/config.yaml \
        /ptmp/bing/2026_sbi_bmode/run89hf_f/config.yaml \
    --test-data \
        /ptmp/bing/2026_sbi_bmode/run91t/data_draws_test.npy \
        /ptmp/bing/2026_sbi_bmode/run91t/data_draws_test.npy \
        /ptmp/bing/2026_sbi_bmode/run91t/data_draws_test.npy \
    --test-params \
        /ptmp/bing/2026_sbi_bmode/run91t/param_draws_test.npy \
        /ptmp/bing/2026_sbi_bmode/run91t/param_draws_test.npy \
        /ptmp/bing/2026_sbi_bmode/run91t/param_draws_test.npy \
    --test-idx 0 \
    --nsamp 10000 \
    --seed 20 \
    --output \
        /ptmp/bing/2026_sbi_bmode/posterior_comparison_1/test.png