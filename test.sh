python scripts/run_sbi_basic.py \
  --odir ./tmp_test_output \
  --config ./scripts/configs/config22.yaml \
  --specdir ./data/ \
  --n_train 100 \
  --n_samples 10000 \
  --n_rounds 1 \
  --pyilcdir ../pyilc \
  --fiducial_beta 1.6 \
  --fiducial_T_dust 19.0 \
  --fiducial_beta_sync -3.1 \
  --deproj_dust \
  --deproj_sync \
  --use_dbeta_map

