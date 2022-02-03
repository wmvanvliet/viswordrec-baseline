"""
Plot correlations between grand-average source level activity and model activity.
"""
import mne
import numpy as np
import pickle
import pandas as pd
from scipy.stats import zscore
from tqdm import tqdm
import os

# Path where you downloaded the OSF data to: https://osf.io/nu2ep/
data_path = './data'

# Whether to overwrite the existing figure and data
overwrite = False

stimuli = pd.read_csv(f'{data_path}/stimuli.csv')
stimuli['type'] = stimuli['type'].astype('category')
stimuli = stimuli.set_index('tif_file')

# Load model layer activations
with open(f'{data_path}/model_layer_activity.pkl', 'rb') as f:
    d = pickle.load(f)

layer_activity = zscore(d['mean_activity'], axis=1)
layer_names = d['layer_names']
layer_activity = [a[stimuli.y] for a in layer_activity]

all_data = np.zeros((len(stimuli), 5124, 60))
counts = np.zeros(len(stimuli))
subjects = range(1, 16)
for subject in tqdm(subjects):
    data, times, vertices_lh, vertices_rh = np.load(f'{data_path}/mne_timecourses/sub-{subject:02d}-mne_timecourses.npz').values()
    metadata = pd.read_csv(f'{data_path}/events/sub-{subject:02d}_task-epasana_events.tsv', sep='\t')
    # Select only relevant trials
    metadata = metadata[metadata.value >= 2]
    metadata = metadata[metadata.value <= 11]
    for tif_file, d in zip(metadata.sort_values('tif_file').tif_file, data):
        y = stimuli.at[tif_file, 'y']
        all_data[y] += d
        counts[y] += 1
ga_data = all_data / counts[:, None, None]

def pearsonr(x, y, *, x_axis=-1, y_axis=-1):
    """Compute Pearson correlation along the given axis."""
    x = zscore(x, axis=x_axis)
    y = zscore(y, axis=y_axis)
    return np.tensordot(x, y, (x_axis, y_axis)) / x.shape[x_axis]

for model_act, layer_name in tqdm(zip(layer_activity, layer_names), total=len(layer_activity)):
    c = mne.SourceEstimate(
        data=pearsonr(ga_data, model_act, x_axis=0, y_axis=0),
        vertices = [vertices_lh, vertices_rh],
        tmin=times[0],
        tstep=times[1] - times[0],
        subject='fsaverage',
    )
    if not os.path.exists(f'{data_path}/brain_model_comparison/brain_model_comparison_mne_{layer_name}-lh.stc') or overwrite:
        os.makedirs(f'{data_path}/brain_model_comparison', exist_ok=True)
        c.save(f'{data_path}/brain_model_comparison/brain_model_comparison_mne_{layer_name}')
