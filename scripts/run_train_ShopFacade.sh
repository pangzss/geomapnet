# mapnet. mask experiment. alpha = 0.5.. no gamma. no sigma.
python train.py --dataset Cambridge \
--scene ShopFacade \
--config_file configs/cambridge_mapnet.ini \
--model mapnet \
--style_dir pbn_train_embedding_dist.txt \
--device 0 \
--init_seed 0