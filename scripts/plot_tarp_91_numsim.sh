python plot_coverage_tarp_main.py \
    --tarpdirs \
        /ptmp/bing/2026_sbi_bmode/tarp90t_b \
        /ptmp/bing/2026_sbi_bmode/tarp89t_obsmat_g \
        /ptmp/bing/2026_sbi_bmode/tarp91hf_c \
        /ptmp/bing/2026_sbi_bmode/tarp91hf_d \
        /ptmp/bing/2026_sbi_bmode/tarp91hf_e \
        /ptmp/bing/2026_sbi_bmode/tarp91hf_f \
        /ptmp/bing/2026_sbi_bmode/tarp91hf_g \
        /ptmp/bing/2026_sbi_bmode/tarp91hf_a \
        /ptmp/bing/2026_sbi_bmode/tarp91hf_b \
    --labels \
        "Train: ObsMat, Test: ObsMat" \
        "Train: Transfer, Test: ObsMat" \
        "MultiFid N=128" \
        "MultiFid N=384" \
        "MultiFid N=768" \
        "MultiFid N=1152" \
        "MultiFid N=1536" \
        "MultiFid N=2048" \
        "MultiFid N=4096" \
    --mode joint \
    --imgdir /ptmp/bing/2026_sbi_bmode/img \
    --name tarp_joint_compare_numsim \
    --title "Multifidelity Joint TARP"