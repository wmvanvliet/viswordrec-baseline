# encoding: utf-8
"""
Construct a dataset containing 256x256 pixel images of rendered words. Uses the
118 words used in the "epasana" study + 127 words used in redness1, padded with
common Finnish words (according to the parsebank) to 5000 words.
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
from multiprocess import Pool, Manager, Process

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Generate the epasana-words dataset')
parser.add_argument('path', type=str, help='The path to write the dataset to.')
parser.add_argument('set', type=str, help='Specify either "train" to generate the training set and to "test" to generate the test set.')
parser.add_argument('-j', '--num-cpus', type=int, default=None, help="Number of CPUs to use. Defaults to all of them.")
parser.add_argument('-n', '--num', type=int, default=1_000_000, help="Total number of stimuli to produce (default: 1M)")
parser.add_argument('-v', '--vocab', type=int, default=10_000, help="Size of the vocabulary (default: 10k)")
parser.add_argument('--noise', action='store_true', help="Add some noise in the background of the images.")
parser.add_argument('--no-experiment', action='store_true', help="Don't include the experiment words in the training set.")
args = parser.parse_args()

# Set this to wherever /m/nbe/scratch/reading_models is mounted on your system
data_path = '/m/nbe/scratch/reading_models'

# The random number generator that will produce all of our randomness.
rng = np.random.RandomState(0)

# Limits
rotations = np.linspace(-20, 20, 11)
sizes = np.linspace(14, 32, 42)
fonts = ['courier new', 'dejavu sans mono', 'times new roman', 'arial',
         'arial black', 'verdana', 'comic sans ms', 'georgia',
         'liberation serif', 'impact', 'roboto condensed']

if args.noise:
    noise_levels = [0.2, 0.3, 0.4]
else:
    noise_levels = [0.0]

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

# Pattern for useful words
pattern = re.compile('^[a-zäö#]+$')

# Read the Finnish Parsebank language model
print('Reading Finnish Parsebank...', flush=True, end='')
vectors = KeyedVectors.load_word2vec_format('/m/nbe/project/corpora/big/parsebank_v4/finnish_parsebank_v4_lemma_5+5.bin', binary=True)
freqs = pd.read_csv('./parsebank_v4_UD_scrambled_conllu_lemmaonly.freq', sep=' ', index_col=0)
freqs['freq'] = freqs['freq'].astype(int)
print('done.')

# Using the "epasana" and "redness1" stimulus lists to select words to plot
epasana = pd.read_csv('stimuli_epasana.csv').query('type=="word"')['text'].str.lower()
redness1 = pd.read_csv('stimuli_redness1.csv', sep=' ', header=None)[0].str.lower()
experiment_words = pd.concat([epasana, redness1], ignore_index=True).drop_duplicates().reset_index(drop=True)
experiment_words = pd.DataFrame(dict(word=experiment_words))
experiment_words['in_corpus'] = experiment_words['word'].copy()
for i, w in enumerate(experiment_words['word'].values):
    if w == 'maalari':
        experiment_words.loc[i, 'in_corpus'] = 'taide#maalari'
    elif w == 'luominen':
        experiment_words.loc[i, 'in_corpus'] = 'luomus'
    elif w == 'oleminen':
        experiment_words.loc[i, 'in_corpus'] = 'olemus'
    elif w == 'eläminen':
        experiment_words.loc[i, 'in_corpus'] = 'elatus'
    elif w == 'koraani':
        experiment_words.loc[i, 'in_corpus'] = 'koraanin'
    elif w == 'huulet':
        experiment_words.loc[i, 'in_corpus'] = 'huuli'
experiment_words = experiment_words.join(freqs, on='in_corpus')
assert len(experiment_words) == 241

# Adding some common finnish words to pad the list to 100_000
more_words = list(vectors.key_to_index.keys())[:100_000]

# Drop words containing capitals (like names) and punctuation characters
more_words = [w for w in more_words if pattern.match(w)]

# Words need to be between 3 and 9 characters (don't count # symbols)
more_words = [w for w in more_words if 2 < len(w.replace('#', '')) <= 9]

more_words = pd.DataFrame(more_words)
more_words.columns = ['word']
more_words['in_corpus'] = more_words['word'].copy()

# Add frequency information and only retain words with high enough frequency
more_words = more_words.join(freqs, on='word')
more_words = more_words.dropna()
more_words = more_words.sort_values('freq', ascending=False).reset_index(drop=True)

words = pd.concat([experiment_words, more_words], ignore_index=True)
words['source'] = ['experiment'] * len(experiment_words) + ['corpus'] * len(more_words)

# Words in the epasana experiment were always shown in upper case
words['stimulus'] = words.word.str.replace('#', '')
words['stimulus'] = words.stimulus.str.upper()

# After our manipulations, we may have some duplicates now
words = words.drop_duplicates('stimulus')

# Remove the epasana words if requested
if args.no_experiment:
    words = words[words['source'] != 'experiment']


# Do we have enough words left after our filters?
assert len(words) >= args.vocab

# Trim word list to 
n_experiment_words = len(words.query('source=="experiment"'))
words = words.iloc[np.hstack((np.arange(n_experiment_words), np.sort(rng.choice(np.arange(n_experiment_words, len(words)), args.vocab - n_experiment_words, replace=False))))]
words = words.reset_index(drop=True)

# Find semantic vectors.
# Some of the epasana words are known in the w2v database under a different
# lemmatization.
vectors_mat = vectors[words['in_corpus']]

# Compute word counts
words['freq_sqrt'] = words.freq ** 0.2
if args.set == 'train':
    multiplier = args.num / (words['freq_sqrt'] / words['freq_sqrt'].min()).sum()
    words['count'] = np.floor(multiplier * words['freq_sqrt'] / words['freq_sqrt'].min()).astype('int')
    n_shards = 200
else:
    words['count'] = 10
    n_shards = 10
n_images = words['count'].sum()
shard_size = int(np.ceil(n_images / n_shards))
print(f'Generating {n_images} images divided over {n_shards} shards of size {shard_size}.')

# Start generating images
rng = np.random.RandomState(0 if args.set == 'train' else 1)
chosen_words = []
chosen_stimuli = []
labels = []
for label, word in words.iterrows():
    for _ in range(word['count']):
        chosen_words.append(word['word'])
        chosen_stimuli.append(word['stimulus'])
        labels.append(label)
shuffled_indices = np.arange(len(chosen_words))
rng.shuffle(shuffled_indices)
chosen_words = np.array(chosen_words)[shuffled_indices]
chosen_stimuli = np.array(chosen_stimuli)[shuffled_indices]
labels = np.array(labels)[shuffled_indices]
chosen_rotations = rng.choice(rotations, n_images)
chosen_sizes = rng.choice(sizes, n_images)
chosen_fonts = rng.choice(list(fonts.keys()), n_images)
chosen_noise_levels = rng.choice(noise_levels, n_images)
stimuli = pd.DataFrame(dict(
    text=chosen_words, stimulus=chosen_stimuli, label=labels,
    rotation=chosen_rotations, size=chosen_sizes, font=chosen_fonts,
    noise=chosen_noise_levels))

makedirs(f'{args.path}/{args.set}', exist_ok=True)

manager = Manager()
queue = manager.Queue()

def track_progress(queue):
    pbar = tqdm(total=n_images)
    for i in range(n_images):
        pbar.update(queue.get())
    pbar.close()

def render_words(shard_num):
    writer = webdataset.TarWriter(f'{args.path}/{args.set}/shard-{shard_num:04d}.tar')

    for key, stimulus in stimuli.iloc[shard_num * shard_size:(shard_num + 1) * shard_size].iterrows():
        buf = render(stimulus['stimulus'], fonts[stimulus['font']][1],
                     stimulus['size'], stimulus['rotation'], stimulus['noise'])

        writer.write({
            '__key__': f'{key}_{stimulus["text"]}',
            'png': bytes(buf),
            'cls': stimulus['label'],
        })
        render_words.queue.put(1)
    writer.close()

def assign_queue_to_producer(queue):
    render_words.queue = queue

progress = Process(target=track_progress, args=(queue,))
progress.start()
print(f'Using {args.num_cpus} CPU cores.')
with Pool(processes=args.num_cpus, initializer=assign_queue_to_producer, initargs=[queue]) as pool:
    pool.map(render_words, range(n_shards))

stimuli.to_csv(f'{args.path}/{args.set}.csv')
pd.DataFrame(vectors_mat, index=words['word']).to_csv(f'{args.path}/vectors.csv')
