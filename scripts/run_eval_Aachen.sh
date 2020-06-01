for h in -0.5 -0.4 -0.3 -0.2 -0.1 0. 0.1 0.2 0.3 0.4 0.5
do
  python eval.py --dataset AachenDayNight --model posenet \
  --init_seed 0 \
  --hue $h \
  --brightness 1 \
  --weights logs/stylized_models/AachenDayNight__mapnet_mapnet_learn_beta_learn_gamma_stylized_16_styles_seed0.pth.tar \
  --config_file configs/style.ini --val
done
