import os
import argparse
import configparser
parser = argparse.ArgumentParser(description='remove checkpoints')
parser.add_argument('--path', type=str)
args = parser.parse_args()


checkpoint_path = args.path#'logs/Cambridge_ShopFacade_mapnet_cambridge_mapnet_100_percent_real_seed0'
files = os.listdir(checkpoint_path)
to_remove = files[:]
print(to_remove)
for f in to_remove:
    suffix = f.split('_')[1]
    print(suffix)