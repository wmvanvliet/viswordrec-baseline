# encoding: utf-8
"""
Construct a dataset containing 256x256 pixel images of rendered words. Uses the
118 words used in the "epasana" study + 82 other frequently used Finnish words.
"""
import argparse
import numpy as np
import pandas as pd
from os import makedirs
from tqdm import tqdm
import webdataset
from gensim.models import KeyedVectors
import re
from render_stimulus import render

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Generate the epasana-words dataset')
parser.add_argument('path', type=str, help='The path to write the dataset to.')
parser.add_argument('set', type=str, help='Specify either "train" to generate the training set and to "test" to generate the test set.')
args = parser.parse_args()

# Set this to wherever /m/nbe/scratch/reading_models is mounted on your system
data_path = '/m/nbe/scratch/reading_models'

# Limits
rotations = np.linspace(-20, 20, 11)
sizes = np.linspace(14, 32, 42)
fonts = ['courier new', 'dejavu sans mono', 'times new roman', 'arial',
         'arial black', 'verdana', 'comic sans ms', 'georgia',
         'liberation serif', 'impact', 'roboto condensed']
noise_levels = [0.1, 0.2, 0.3]

fonts = {
    'ubuntu mono': [None, f'{data_path}/fonts/UbuntuMono-R.ttf'],
    'courier': [None, f'{data_path}/fonts/courier.ttf'],
    'luxi mono regular': [None, f'{data_path}/fonts/luximr.ttf'],
    'lucida console': [None, f'{data_path}/fonts/LucidaConsole-R.ttf'],
    'lekton': [None, f'{data_path}/fonts/Lekton-Regular.ttf'],
    'dejavu sans mono': [None, f'{data_path}/fonts/DejaVuSansMono.ttf'],
    'times new roman': [None, f'{data_path}/fonts/times.ttf'],
    'arial': [None, f'{data_path}/fonts/arial.ttf'],
    'arial black': [None, f'{data_path}/fonts/arialbd.ttf'],
    'verdana': [None, f'{data_path}/fonts/verdana.ttf'],
    'comic sans ms': [None, f'{data_path}/fonts/comic.ttf'],
    'georgia': [None, f'{data_path}/fonts/georgia.ttf'],
    'liberation serif': [None, f'{data_path}/fonts/LiberationSerif-Regular.ttf'],
    'impact': [None, f'{data_path}/fonts/impact.ttf'],
    'roboto condensed': [None, f'{data_path}/fonts/Roboto-Light.ttf'],
}

# Using the "epasana" stimulus list to select words to plot
words = pd.read_csv('/m/nbe/scratch/epasana/stimuli.csv').query('type=="word"')['text']
words = words.str.lower()

# Read the Finnish Parsebank language model
print('Reading Finnish Parsebank...', flush=True, end='')
vectors = KeyedVectors.load_word2vec_format('/m/nbe/project/corpora/big/parsebank_v4/finnish_parsebank_v4_lemma_5+5.bin', binary=True)
print('done.')

# Adding some common finnish words to pad the list to 60_000
more_words = list(vectors.key_to_index.keys())[:1_000_000]

# Drop words containing capitals (like names) and punctuation characters
pattern = re.compile('^[a-zäö#]+$')
more_words = [w for w in more_words if pattern.match(w)]

# Words need to be at least length 2 (don't count # symbols)
more_words = [w for w in more_words if len(w.replace('#', '')) >= 2]

# Do we have enough words left after our filters?
assert len(more_words) >= 10_000

# Pad the original word list up to 60_000 words
more_words = pd.DataFrame(more_words)
words = pd.concat([words, more_words], ignore_index=True)
words = words.drop_duplicates()

# Some of the epasana words are known in the w2v database under a different
# lemmatization.
words_in_w2v = words.copy()
for i, w in enumerate(words.values):
    if w == 'maalari':
        words_in_w2v.iloc[i] = 'taide#maalari'
    elif w == 'luominen':
        words_in_w2v.iloc[i] = 'luomus'
    elif w == 'oleminen':
        words_in_w2v.iloc[i] = 'olemus'
    elif w == 'eläminen':
        words_in_w2v.iloc[i] = 'elatus'
    elif w == 'koraani':
        words_in_w2v.iloc[i] = 'koraanin'
# Perform the lookup for the w2v vectors for each chosen word
vectors = vectors[words_in_w2v[0]]

# Words in the epasana experiment were always shown in upper case
words = pd.Series([word.upper() for word in words[0]])

# Drop # signs deliminating compound words
words = words.str.replace('#', '')

# After our manipulations, we may have some duplicates now
words = words.drop_duplicates()

# Select 10k most common words
words = words[:10_000]
vectors = vectors[words.index]

# Start generating images
rng = np.random.RandomState(0)

chosen_rotations = []
chosen_sizes = []
chosen_fonts = []
chosen_words = []
chosen_noise_levels = []

n = 100 if args.set == 'train' else 10
labels = np.zeros(len(words) * n, dtype=int)

makedirs(f'{args.path}/{args.set}', exist_ok=True)
writer = webdataset.ShardWriter(f'{args.path}/{args.set}/shard-%04d.tar',
                                maxcount=10_000)

pbar = tqdm(total=n * len(words))
for i in range(n):
    for label, word in enumerate(words):
        word = word.replace('#', '')

        rotation = rng.choice(rotations)
        fontsize = rng.choice(sizes)
        font = rng.choice(list(fonts.keys()))
        noise_level = rng.choice(noise_levels)

        buf = render(word, fonts[font][1], fontsize, rotation, noise_level);

        writer.write({
            '__key__': f'{word}{i:05d}',
            'jpeg': bytes(buf),
            'cls': label,
        })

        chosen_words.append(word)
        chosen_rotations.append(rotation)
        chosen_sizes.append(fontsize)
        chosen_fonts.append(font)
        chosen_noise_levels.append(noise_level)

        labels[i * len(words) + label] = label
        pbar.update(1)
writer.close()
pbar.close()

df = pd.DataFrame(dict(text=chosen_words, rotation=chosen_rotations, noise=chosen_noise_levels,
                       size=chosen_sizes, font=chosen_fonts, label=labels))
df.to_csv(f'{args.path}/{args.set}.csv')
pd.DataFrame(vectors, index=words).to_csv(f'{args.path}/vectors.csv')
