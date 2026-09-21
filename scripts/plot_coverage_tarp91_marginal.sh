python plot_coverage_tarp_main.py \
    --tarpdirs \
        /ptmp/bing/2026_sbi_bmode/tarp89t_g \
        /ptmp/bing/2026_sbi_bmode/tarp89t_obsmat_g \
        /ptmp/bing/2026_sbi_bmode/tarp91hf_a \
    --labels \
        "Original" \
        "ObsMat" \
        "MultiFid" \
    --mode marginal \
    --imgdir /ptmp/bing/2026_sbi_bmode/tarp91hf_a/img \
    --name tarp_marginal_compare \
    --title "Multifidelity N = 2048"