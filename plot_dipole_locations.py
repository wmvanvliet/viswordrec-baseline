"""
Make figures of the left hemisphere of the brain, with MNE activation and the
locations of the selected 3 groups of dipoles. This forms the top part of the
main results figure in the paper.
"""
import mne
from config import fname, subjects
from mayavi import mlab
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_path = './data'

# The brain will be plotted from two angles.
views = [
    (180, 90, 440, [0, 0, 0]),
    (-90, 180, 440, [0, 0, 0]),
]
view_names = ['lateral', 'ventral']

##
# Read in the dipoles in Talairach coordinates. The resulting positions will be
# placed on the brain figure as "foci": little spheres.
foci = []
for subject in subjects:
    dips = mne.read_dipole(f'{data_path}/dipoles/sub-{subject:02d}_task-epasana_dipoles_talairach.bdip')
    dip_selection = pd.read_csv(f'{data_path}/dipoles/sub-{subject:02d}_task-epasana_dipole_selection.tsv', sep='\t', index_col=0)
    pos_tal = dips.pos * 1000
    if 'LeftOcci1' in dip_selection.index:
        foci.append(pos_tal[dip_selection.loc['LeftOcci1'].dipole])
foci = np.array(foci)

# The MNE activation to valid words will be plotted on the brain as well.
ga_stc = mne.read_source_estimate(f'{data_path}/grand_average/grand_average-word')


## Dipole group 1 (Early visual response)
fig1 = mlab.figure(1, size=(1000, 1000))
mne.viz.set_3d_backend('mayavi')
brain = ga_stc.copy().crop(0.065, 0.11).mean().plot(
    'fsaverage',
    subjects_dir=fname.subjects_dir,
    hemi='lh',
    background='white',
    cortex='low_contrast',
    surface='inflated',
    figure=fig1,
)
brain.scale_data_colormap(3.5, 3.8, 7, True)

# Save images without dipoles
for view, view_name in zip(views, view_names):
    mlab.view(*view)
    # Save with a transparent background (must route through mlab.screenshot to obtain this)
    plt.imsave(f'figures/landmark1_{view_name}_no_dipoles.png', mlab.screenshot(mode='rgba', antialiased=True))

# Save images with dipoles
brain.add_foci(foci, map_surface='white', hemi='lh', scale_factor=0.2)
for view, view_name in zip(views, view_names):
    mlab.view(*view)
    plt.imsave(f'figures/landmark1_{view_name}_with_dipoles.png', mlab.screenshot(mode='rgba', antialiased=True))


## Dipole group 2 (Letter-string response)
foci = []
for subject in subjects:
    dips = mne.read_dipole(f'{data_path}/dipoles/sub-{subject:02d}_task-epasana_dipoles_talairach.bdip')
    dip_selection = pd.read_csv(f'{data_path}/dipoles/sub-{subject:02d}_task-epasana_dipole_selection.tsv', sep='\t', index_col=0)
    pos_tal = dips.pos * 1000
    if 'LeftOcciTemp2' in dip_selection.index:
        foci.append(pos_tal[dip_selection.loc['LeftOcciTemp2'].dipole])
foci = np.array(foci)
fig2 = mlab.figure(2, size=(1000, 1000))
brain = ga_stc.copy().crop(0.14, 0.2).mean().plot(
    'fsaverage',
    subjects_dir=fname.subjects_dir,
    hemi='lh',
    background='white',
    cortex='low_contrast',
    surface='inflated',
    figure=fig2,
)
brain.scale_data_colormap(3.5, 3.8, 7, True)

# Save images without dipoles
for view, view_name in zip(views, view_names):
    mlab.view(*view)
    # Save with a transparent background (must route through mlab.screenshot to obtain this)
    plt.imsave(f'figures/landmark2_{view_name}_no_dipoles.png', mlab.screenshot(mode='rgba', antialiased=True))

# Save images with dipoles
brain.add_foci(foci, map_surface='white', hemi='lh', scale_factor=0.2)
for view, view_name in zip(views, view_names):
    mlab.view(*view)
    plt.imsave(f'figures/landmark2_{view_name}_with_dipoles.png', mlab.screenshot(mode='rgba', antialiased=True))


## Dipole group 3 (N400 response)
foci = []
for subject in subjects:
    dips = mne.read_dipole(f'{data_path}/dipoles/sub-{subject:02d}_task-epasana_dipoles_talairach.bdip')
    dip_selection = pd.read_csv(f'{data_path}/dipoles/sub-{subject:02d}_task-epasana_dipole_selection.tsv', sep='\t', index_col=0)
    pos_tal = dips.pos * 1000
    if 'LeftTemp3' in dip_selection.index:
        foci.append(pos_tal[dip_selection.loc['LeftTemp3'].dipole])
foci = np.array(foci)
fig = mlab.figure(3, size=(1000, 1000))
brain = ga_stc.copy().crop(0.3, 0.5).mean().plot(
    'fsaverage',
    subjects_dir=fname.subjects_dir,
    hemi='lh',
    background='white',
    cortex='low_contrast',
    surface='inflated',
    figure=fig,
)
brain.scale_data_colormap(3.5, 3.8, 7, True)

# Save images without dipoles
for view, view_name in zip(views, view_names):
    mlab.view(*view)
    # Save with a transparent background (must route through mlab.screenshot to obtain this)
    plt.imsave(f'figures/landmark3_{view_name}_no_dipoles.png', mlab.screenshot(mode='rgba', antialiased=True))

# Save images with dipoles
brain.add_foci(foci, map_surface='white', hemi='lh', scale_factor=0.2)
for view, view_name in zip(views, view_names):
    mlab.view(*view)
    plt.imsave(f'figures/landmark3_{view_name}_with_dipoles.png', mlab.screenshot(mode='rgba', antialiased=True))
