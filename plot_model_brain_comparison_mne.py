"""
Plot correlations between grand-average source level activity and model activity.
"""
import mne
import numpy as np
import pandas as pd
import os

# Path to the OSF downloaded data
data_path = './data'

stimuli = pd.read_csv(f'{data_path}/stimuli.csv')
stimuli['type'] = stimuli['type'].astype('category')

# The brain will be plotted from two angles.
views = ['lateral', 'ventral']
map_surface = 'white'

layer_names = ['conv1_relu', 'conv2_relu', 'conv3_relu', 'conv4_relu', 'conv5_relu', 'fc1_relu', 'fc2_relu', 'word_relu']
os.makedirs('figures', exist_ok=True)

## Layers 1-5 vs dipole group 1 (Early visual response)
for layer_name in layer_names[:5]:
    c = mne.read_source_estimate(f'{data_path}/model_brain_comparison/brain_model_comparison_mne_{layer_name}')
    c = c.copy().crop(0.070, 0.130)
    c.data = np.maximum(c.data, 0)
    brain = c.plot(
        'fsaverage',
        subjects_dir=data_path,
        hemi='lh',
        background='white',
        cortex='low_contrast',
        surface='inflated',
        initial_time = c.get_peak()[1],
        clim=dict(kind='value', lims=(0.2, 0.55, 0.9)),
        time_viewer=False,
    )

    # Save images
    for view in views:
        brain.show_view(view)
        brain.save_image(f'figures/{layer_name}_{view}_mne.png')

## Layers 6-7 vs dipole group 2 (Letter string response)
for layer_name in layer_names[5:7]:
    c = mne.read_source_estimate(f'{data_path}/model_brain_comparison/brain_model_comparison_mne_{layer_name}')
    c = c.copy().crop(0.115, 0.200)
    c.data = np.maximum(c.data, 0)
    brain = c.plot(
        'fsaverage',
        subjects_dir=data_path,
        hemi='lh',
        background='white',
        cortex='low_contrast',
        surface='inflated',
        initial_time = c.get_peak()[1],
        clim=dict(kind='value', lims=(0.1, 0.225, 0.35)),
        time_viewer=False,
    )

    # Save images
    for view in views:
        brain.show_view(view)
        brain.save_image(f'figures/{layer_name}_{view}_mne.png')

## Layer 8 vs dipole group 3 (N400m response)
layer_name = layer_names[7]
c = mne.read_source_estimate(f'{data_path}/model_brain_comparison/brain_model_comparison_mne_{layer_name}')
c = c.copy().crop(0.300, 0.400)
c.data = np.maximum(c.data, 0)
brain = c.plot(
    'fsaverage',
    subjects_dir=data_path,
    hemi='lh',
    background='white',
    cortex='low_contrast',
    surface='inflated',
    initial_time = c.get_peak()[1],
    clim=dict(kind='value', lims=(0.05, 0.125, 0.20)),
    time_viewer=False,
)

# Save images
for view in views:
    brain.show_view(view)
    brain.save_image(f'figures/{layer_name}_{view}_mne_no.png')
