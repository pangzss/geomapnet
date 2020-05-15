# mapnet. mask experiment. alpha = 0.5.. no gamma. no sigma.
python train.py --dataset Cambridge \
--scene ShopFacade \
--config_file configs/cambridge_mapnet.ini \
--model mapnet \
--style_dir ../data/pbn_test_embedding_dist.txt \
--alpha 1.0 \
--real_prob 0 \
--mask \
--device 0 \
--init_seed 0 \
--suffix _mask




