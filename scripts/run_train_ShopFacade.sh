# mapnet. mask experiment. alpha = 0.5.. no gamma. no sigma.
python train.py --dataset Cambridge \
--scene ShopFacade \
--config_file configs/cambridge_SLocNet.ini \
--model SLocNet \
--style_dir ../data/style_selected \
--device 0 \
--init_seed 0 \
--learn_beta \
--learn_gamma \
--learn_sigma




