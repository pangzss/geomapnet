# kings college. mapnet. No style. Learn beta. learn gamma.
python train.py --dataset Cambridge \
--scene KingsCollege \
--config_file configs/cambridge_mapnet.ini \
--model mapnet \
--style_dir ../data/pbn_train_embedding_dist.txt \
--device 0 \
--init_seed 0 \
--learn_beta \
--learn_gamma 

python train.py --dataset Cambridge \
--scene KingsCollege \
--config_file configs/cambridge_mapnet.ini \
--model mapnet \
--style_dir ../data/pbn_train_embedding_dist.txt \
--real_prob 50 \
--alpha 0.5 \
--device 0 \
--init_seed 0 \
--learn_beta \
--learn_gamma 
