# old hospital. mapnet. No style. Learn beta. learn gamma.
python train.py --dataset Cambridge \
--scene OldHospital \
--config_file configs/cambridge_mapnet.ini \
--model mapnet \
--style_dir ../data/pbn_train_embedding_dist.txt \
--device 0 \
--init_seed 1 \
--checkpoint logs/Cambridge/Cambridge_OldHospital_mapnet_cambridge_mapnet_100_percent_real_alpha1.0_seed1/epoch_060.pth.tar \
--resume_optim


