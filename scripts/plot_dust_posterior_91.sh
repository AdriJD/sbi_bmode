#!/bin/bash

source /u/bing/sbi_bmode/.venv/bin/activate

python /u/bing/sbi_bmode/scripts/plot_posterior_compare.py \
    --posteriors \
        /ptmp/bing/2026_sbi_bmode/run89_optuna_g/trial_0133/posterior.pkl \
        /ptmp/bing/2026_sbi_bmode/run91hf_optuna_c/trial_0078/posterior.pkl \
        /ptmp/bing/2026_sbi_bmode/run91hf_optuna_d/trial_0076/posterior.pkl \
        /ptmp/bing/2026_sbi_bmode/run91hf_optuna_e/trial_0105/posterior.pkl \
        /ptmp/bing/2026_sbi_bmode/run91hf_optuna_f/trial_0092/posterior.pkl \
        /ptmp/bing/2026_sbi_bmode/run91hf_optuna_g/trial_0078/posterior.pkl \
        /ptmp/bing/2026_sbi_bmode/run91hf_optuna_a/trial_0138/posterior.pkl \
        /ptmp/bing/2026_sbi_bmode/run90_optuna_b/trial_0032/posterior.pkl \
    --labels \
        '$\mathrm{Transfer},\ N_{\mathrm{transfer}}=20480$' \
        '$\mathrm{MultiFid},\ N_{\mathrm{transfer}}=20480,\ N_{\mathrm{obsmat}}=128$' \
        '$\mathrm{MultiFid},\ N_{\mathrm{transfer}}=20480,\ N_{\mathrm{obsmat}}=384$' \
        '$\mathrm{MultiFid},\ N_{\mathrm{transfer}}=20480,\ N_{\mathrm{obsmat}}=768$' \
        '$\mathrm{MultiFid},\ N_{\mathrm{transfer}}=20480,\ N_{\mathrm{obsmat}}=1152$' \
        '$\mathrm{MultiFid},\ N_{\mathrm{transfer}}=20480,\ N_{\mathrm{obsmat}}=1536$' \
        '$\mathrm{MultiFid},\ N_{\mathrm{transfer}}=20480,\ N_{\mathrm{obsmat}}=2048$' \
        '$\mathrm{ObsMat},\ N_{\mathrm{obsmat}}=20480$' \
    --configs \
        /ptmp/bing/2026_sbi_bmode/run89t_obsmat_g/config.yaml \
        /ptmp/bing/2026_sbi_bmode/run91hf_c/config.yaml \
        /ptmp/bing/2026_sbi_bmode/run91hf_d/config.yaml \
        /ptmp/bing/2026_sbi_bmode/run91hf_e/config.yaml \
        /ptmp/bing/2026_sbi_bmode/run91hf_f/config.yaml \
        /ptmp/bing/2026_sbi_bmode/run91hf_g/config.yaml \
        /ptmp/bing/2026_sbi_bmode/run91hf_a/config.yaml \
        /ptmp/bing/2026_sbi_bmode/run90t_b/config.yaml \
    --test-data \
        /ptmp/bing/2026_sbi_bmode/run89t_obsmat_g/data_draws_test.npy \
        /ptmp/bing/2026_sbi_bmode/run89t_obsmat_g/data_draws_test.npy \
        /ptmp/bing/2026_sbi_bmode/run89t_obsmat_g/data_draws_test.npy \
        /ptmp/bing/2026_sbi_bmode/run89t_obsmat_g/data_draws_test.npy \
        /ptmp/bing/2026_sbi_bmode/run89t_obsmat_g/data_draws_test.npy \
        /ptmp/bing/2026_sbi_bmode/run89t_obsmat_g/data_draws_test.npy \
        /ptmp/bing/2026_sbi_bmode/run89t_obsmat_g/data_draws_test.npy \
        /ptmp/bing/2026_sbi_bmode/run90t_b/data_draws_test.npy \
    --test-params \
        /ptmp/bing/2026_sbi_bmode/run89t_obsmat_g/param_draws_test.npy \
        /ptmp/bing/2026_sbi_bmode/run89t_obsmat_g/param_draws_test.npy \
        /ptmp/bing/2026_sbi_bmode/run89t_obsmat_g/param_draws_test.npy \
        /ptmp/bing/2026_sbi_bmode/run89t_obsmat_g/param_draws_test.npy \
        /ptmp/bing/2026_sbi_bmode/run89t_obsmat_g/param_draws_test.npy \
        /ptmp/bing/2026_sbi_bmode/run89t_obsmat_g/param_draws_test.npy \
        /ptmp/bing/2026_sbi_bmode/run89t_obsmat_g/param_draws_test.npy \
        /ptmp/bing/2026_sbi_bmode/run90t_b/param_draws_test.npy \
    --test-idx 1 \
    --nsamp 10000 \
    --seed 20 \
    --gradient \
    --param-idx 0 2 3 \
    --output \
        /ptmp/bing/2026_sbi_bmode/posterior_comparison/test_dust.png