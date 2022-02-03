"""
Plot correlations between grand-average source level activity and model activity.
"""
import mne
import numpy as np
import pickle
import pandas as pd
from mayavi import mlab
import matplotlib.pyplot as plt
from scipy.stats import zscore
import os

# Path to the OSF downloaded data
data_path = './data'

stimuli = pd.read_csv(f'{data_path}/stimuli.csv')
stimuli['type'] = stimuli['type'].astype('category')

## Load model layer activations
with open(f'{data_path}/model_layer_activity.pkl', 'rb') as f:
    d = pickle.load(f)

layer_activity = zscore(d['mean_activity'], axis=1)
layer_names = d['layer_names']
layer_acts = pd.DataFrame(layer_activity.T, columns=layer_names)

# The brain will be plotted from two angles.
views = [
    (180, 90, 440, [0, 0, 0]),
    (-90, 180, 440, [0, 0, 0]),
]
view_names = ['lateral', 'ventral']

##
# Read in the dipoles and morph their positions to the fsaverage brain.
# The resulting positions will be placed on the brain figure as "foci": little
# spheres.
foci1 = []
foci2 = []
foci3 = []
subjects = list(range(1, 16))
for subject in subjects:
    dips_tal = mne.read_dipole(f'{data_path}/dipoles/sub-{subject:02d}_task-epasana_dipoles_talairach.bdip')
    dip_selection = pd.read_csv(f'{data_path}/dipoles/sub-{subject:02d}_task-epasana_dipole_selection.tsv', sep='\t', index_col=0)
    pos_tal = dips_tal.pos * 1000
    if 'LeftOcci1' in dip_selection.index:
        foci1.append(pos_tal[dip_selection.loc['LeftOcci1'].dipole])
    if 'LeftOcciTemp2' in dip_selection.index:
        foci2.append(pos_tal[dip_selection.loc['LeftOcciTemp2'].dipole])
    if 'LeftTemp3' in dip_selection.index:
        foci3.append(pos_tal[dip_selection.loc['LeftTemp3'].dipole])
foci1 = np.array(foci1)
foci2 = np.array(foci2)
foci3 = np.array(foci3)

mne.viz.set_3d_backend('mayavi')
os.makedirs('figures', exist_ok=True)

## Layers 1-5 vs dipole group 1 (Early visual response)
for layer_name in layer_names[:5]:
    c = mne.read_source_estimate(f'{data_path}/brain_model_comparison/brain_model_comparison_mne_{layer_name}')
    c = c.copy().crop(0.065, 0.115)
    c.data = np.maximum(c.data, 0)
    fig = mlab.figure(size=(1000, 1000))
    brain = c.plot(
        'fsaverage',
        subjects_dir=data_path,
        hemi='lh',
        background='white',
        cortex='low_contrast',
        surface='inflated',
        figure=fig,
        initial_time = c.get_peak()[1],
    )
    brain.scale_data_colormap(0.2, 0.55, 0.9, True)

    # Save images without dipoles
    for view, view_name in zip(views, view_names):
        mlab.view(*view)
        plt.imsave(f'figures/{layer_name}_{view_name}_mne_no_dipoles.png', mlab.screenshot(mode='rgba', antialiased=True))

    # Save images with dipoles
    brain.add_foci(foci1, map_surface='white', hemi='lh', scale_factor=0.2)
    for view, view_name in zip(views, view_names):
        mlab.view(*view)
        plt.imsave(f'figures/{layer_name}_{view_name}_mne_with_dipoles.png', mlab.screenshot(mode='rgba', antialiased=True))

## Layers 6-7 vs dipole group 2 (Letter string response)
for layer_name in layer_names[5:7]:
    c = mne.read_source_estimate(f'{data_path}/brain_model_comparison/brain_model_comparison_mne_{layer_name}')
    c = c.copy().crop(0.14, 0.2)
    c.data = np.maximum(c.data, 0)
    fig = mlab.figure(size=(1000, 1000))
    brain = c.plot(
        'fsaverage',
        subjects_dir=data_path,
        hemi='lh',
        background='white',
        cortex='low_contrast',
        surface='inflated',
        figure=fig,
        initial_time = c.get_peak()[1],
    )
    brain.scale_data_colormap(0.1, 0.225, 0.35, True)

    # Save images without dipoles
    for view, view_name in zip(views, view_names):
        mlab.view(*view)
        plt.imsave(f'figures/{layer_name}_{view_name}_mne_no_dipoles.png', mlab.screenshot(mode='rgba', antialiased=True))

    # Save images with dipoles
    brain.add_foci(foci2, map_surface='white', hemi='lh', scale_factor=0.2)
    for view, view_name in zip(views, view_names):
        mlab.view(*view)
        plt.imsave(f'figures/{layer_name}_{view_name}_mne_with_dipoles.png', mlab.screenshot(mode='rgba', antialiased=True))

## Layer 8 vs dipole group 3 (N400m response)
layer_name = layer_names[7]
c = mne.read_source_estimate(f'{data_path}/brain_model_comparison/brain_model_comparison_mne_{layer_name}')
c = c.copy().crop(0.3, 0.5)
c.data = np.maximum(c.data, 0)
fig = mlab.figure(size=(1000, 1000))
brain = c.plot(
    'fsaverage',
    subjects_dir=data_path,
    hemi='lh',
    background='white',
    cortex='low_contrast',
    surface='inflated',
    figure=fig,
    initial_time = c.get_peak()[1],
)
brain.scale_data_colormap(0.05, 0.125, 0.20, True)

# Save images without dipoles
for view, view_name in zip(views, view_names):
    mlab.view(*view)
    plt.imsave(f'figures/{layer_name}_{view_name}_mne_no_dipoles.png', mlab.screenshot(mode='rgba', antialiased=True))

# Save images with dipoles
brain.add_foci(foci3, map_surface='white', hemi='lh', scale_factor=0.2)
for view, view_name in zip(views, view_names):
    mlab.view(*view)
    plt.imsave(f'figures/{layer_name}_{view_name}_mne_with_dipoles.png', mlab.screenshot(mode='rgba', antialiased=True))
