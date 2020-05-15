# old hospital. mapnet. No style. Learn beta. learn gamma.
python train.py --dataset Cambridge \
--scene OldHospital \
--config_file configs/cambridge_trinet_pc.ini \
--model trinet \
--style_dir ../data/pbn_train_embedding_dist.txt \
--device 0 \
--init_seed 0 \
--checkpoint logs/Cambridge/Cambridge_OldHospital_mapnet_OldHospital_mapnet_50_percent_real_beta_gamma_alpha0.5_seed0_version_1/epoch_360.pth.tar


