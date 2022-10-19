"""
Construct a dataset containing 128x128 pixel images of random pixels.
"""
# encoding: utf-8
import argparse
import numpy as np
import pandas as pd
from os import makedirs
from tqdm import tqdm
import webdataset
from render_stimulus import render


# Parse command-line arguments
parser = argparse.ArgumentParser(description='Generate the noise dataset')
parser.add_argument('path', type=str, help='The path to write the dataset to.')
parser.add_argument('set', type=str, help='Specify either "train" to generate the training set and to "test" to generate the test set.')
args = parser.parse_args()

data_path = '/m/nbe/scratch/reading_models'
font = f'{data_path}/fonts/UbuntuMono-R.ttf'

noise_levels = [0.1, 0.2, 0.3]

rng = np.random.RandomState(0)

# Render the consonant strings as PNG files
chosen_noise_levels = []

n = 50_000 if args.set == 'train' else 5_000

makedirs(f'{args.path}/{args.set}', exist_ok=True)
writer = webdataset.ShardWriter(f'{args.path}/{args.set}/shard-%04d.tar',
                                maxcount=1_000)

for i in tqdm(range(n), total=n):
    noise_level = rng.choice(noise_levels)
    buf = render('', font, 0, 0, noise_level);
    chosen_noise_levels.append(noise_level)

    writer.write({
        '__key__': f'{i:06d}',
        'jpeg': bytes(buf),
        'cls': 0,
    })
writer.close()

df = pd.DataFrame(dict(noise=chosen_noise_levels, label=np.zeros(n)))
df['text'] = 'noise'
df.to_csv(f'{args.path}/{args.set}.csv')
pd.DataFrame(np.zeros((1, 300))).to_csv(f'{args.path}/vectors.csv')
