python train.py --dataset AachenDayNight \
--config_file configs/style.ini \
--model posenet \
--style_dir ../data/style_selected \
--real_prob 90 \
--alpha 1.0 \
--device 0 \
--init_seed 0 \
--learn_beta \
--learn_gamma 
#--checkpoint logs/AachenDayNight_posenet_style_4_styles_50_percent_real_seed0/epoch_050.pth.tar \
#--resume_optim
