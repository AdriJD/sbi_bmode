python plot_numsim_loss.py \
    --optdirs \
        /ptmp/bing/2026_sbi_bmode/run91hf_optuna_c \
        /ptmp/bing/2026_sbi_bmode/run91hf_optuna_d \
        /ptmp/bing/2026_sbi_bmode/run91hf_optuna_e \
        /ptmp/bing/2026_sbi_bmode/run91hf_optuna_f \
        /ptmp/bing/2026_sbi_bmode/run91hf_optuna_g \
        /ptmp/bing/2026_sbi_bmode/run91hf_optuna_a \
        /ptmp/bing/2026_sbi_bmode/run91hf_optuna_b \
        /ptmp/bing/2026_sbi_bmode/run90_optuna_b \
    --nsims \
        128 \
        384 \
        768 \
        1152 \
        1536 \
        2048 \
        4096 \
        20480 \
    --imgdir /ptmp/bing/2026_sbi_bmode/img \
    --name loss_vs_nhf \
    --title "Multifidelity training"