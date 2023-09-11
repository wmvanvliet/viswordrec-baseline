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

# Model to do the computations for
model_name = 'vgg11stochastic_first_imagenet_then_10kwords-freq'

# Whether to overwrite the existing figure and data
overwrite = False

stimuli = pd.read_csv(f'{data_path}/stimuli.csv')
stimuli['type'] = stimuli['type'].astype('category')
stimuli = stimuli.set_index('tif_file')

# Load model layer activations
with open(f'{data_path}/model_layer_activity/{model_name}_mean_layer_activity_torch19.pkl', 'rb') as f:
    d = pickle.load(f)

layer_activity = zscore(d['mean_activity'], axis=1)
layer_names = d['layer_names']

data, vertices_lh, vertices_rh, times, subject = np.load(f'{data_path}/grand_average/grand_average_mne_timecourses.npz').values()

def pearsonr(x, y, *, x_axis=-1, y_axis=-1):
    """Compute Pearson correlation along the given axis."""
    x = zscore(x, axis=x_axis)
    y = zscore(y, axis=y_axis)
    return np.tensordot(x, y, (x_axis, y_axis)) / x.shape[x_axis]

for model_act, layer_name in tqdm(zip(layer_activity, layer_names), total=len(layer_activity)):
    c = mne.SourceEstimate(
        data=pearsonr(data, model_act, x_axis=0, y_axis=0),
        vertices = [vertices_lh, vertices_rh],
        tmin=times[0],
        tstep=times[1] - times[0],
        subject='fsaverage',
    )
    if not os.path.exists(f'{data_path}/model_brain_comparison/model_brain_comparison_mne_{layer_name}-lh.stc') or overwrite:
        os.makedirs(f'{data_path}/model_brain_comparison', exist_ok=True)
        c.save(f'{data_path}/model_brain_comparison/model_brain_comparison_mne_{layer_name}')
