python train.py --dataset AachenDayNight \
--config_file configs/style.ini \
--model posenet \
--num_styles 4 \
--device 0 \
--learn_beta \
--learn_gamma \
--checkpoint logs/AachenDayNight_posenet_style_4_styles_40_percent_real_seed0/epoch_300.pth.tar \
--resume_optim
