# mapnet. 50. 0.5 . Nan. . beta, gamma,.
python train.py --dataset AachenDayNight \
--config_file configs/Aachen_mapnet.ini \
--model mapnet \
--style_dir ../data/pbn_train_embedding_dist.txt \
--alpha 0.5 \
--device 0 \
--real_prob 50 \
--init_seed 0  \
--learn_beta \
--learn_gamma

