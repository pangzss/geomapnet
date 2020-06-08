
python eval.py --dataset Cambridge --scene ShopFacade --model posenet \
--init_seed 0 \
--weights logs/Cambridge/Cambridge_ShopFacade_mapnet_50_percent_real_alpha1.0_seed3_add_noise/epoch_500.pth.tar \
--config_file configs/cambridge_mapnet.ini --val

