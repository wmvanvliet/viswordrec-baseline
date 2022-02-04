"""
Make figures of the left hemisphere of the brain, with MNE activation and the
locations of the selected 3 groups of dipoles. This forms the top part of the
main results figure in the paper.
"""
import mne
from mayavi import mlab
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data_path = './data'


# The brain will be plotted from two angles.
views = ['lateral', 'ventral']
#map_surface = 'white'
map_surface = None

##
# Read in the dipoles in Talairach coordinates. The resulting positions will be
# placed on the brain figure as "foci": little spheres.
foci = []
subjects = list(range(1, 16))
for subject in subjects:
    dips = mne.read_dipole(f'{data_path}/dipoles/sub-{subject:02d}/meg/sub-{subject:02d}_task-epasana_dipoles_talairach.bdip')
    dip_selection = pd.read_csv(f'{data_path}/dipoles/sub-{subject:02d}/meg/sub-{subject:02d}_task-epasana_dipole_selection.tsv', sep='\t', index_col=0)
    pos_tal = dips.pos * 1000
    if 'LeftOcci1' in dip_selection.index:
        foci.append(pos_tal[dip_selection.loc['LeftOcci1'].dipole])
foci = np.array(foci)

# The MNE activation to valid words will be plotted on the brain as well.
ga_stc = mne.read_source_estimate(f'{data_path}/grand_average/grand_average-word')


## Dipole group 1 (Early visual response)
brain = ga_stc.copy().crop(0.065, 0.11).mean().plot(
    'fsaverage',
    subjects_dir=data_path,
    hemi='lh',
    background='white',
    cortex='low_contrast',
    surface='inflated',
    clim=dict(kind='value', lims=(3.5, 3.8, 7)),
    brain_kwargs=dict(show_toolbar=False),
)

# Save images without dipoles
for view in views:
    brain.show_view(view)
    brain.save_image(f'figures/landmark1_{view}_no_dipoles.png')

# Save images with dipoles
brain.add_foci(foci, map_surface=map_surface, hemi='lh', scale_factor=0.2)
for view in views:
    brain.show_view(view)
    brain.save_image(f'figures/landmark1_{view}_with_dipoles.png')


## Dipole group 2 (Letter-string response)
foci = []
for subject in subjects:
    dips = mne.read_dipole(f'{data_path}/dipoles/sub-{subject:02d}/meg/sub-{subject:02d}_task-epasana_dipoles_talairach.bdip')
    dip_selection = pd.read_csv(f'{data_path}/dipoles/sub-{subject:02d}/meg/sub-{subject:02d}_task-epasana_dipole_selection.tsv', sep='\t', index_col=0)
    pos_tal = dips.pos * 1000
    if 'LeftOcciTemp2' in dip_selection.index:
        foci.append(pos_tal[dip_selection.loc['LeftOcciTemp2'].dipole])
foci = np.array(foci)
brain = ga_stc.copy().crop(0.14, 0.2).mean().plot(
    'fsaverage',
    subjects_dir=data_path,
    hemi='lh',
    background='white',
    cortex='low_contrast',
    surface='inflated',
    clim=dict(kind='value', lims=(3.5, 3.8, 7)),
    brain_kwargs=dict(show_toolbar=False),
)

# Save images without dipoles
for view in views:
    brain.show_view(view)
    brain.save_image(f'figures/landmark2_{view}_no_dipoles.png')

# Save images with dipoles
brain.add_foci(foci, map_surface=map_surface, hemi='lh', scale_factor=0.2)
for view in views:
    brain.show_view(view)
    brain.save_image(f'figures/landmark2_{view}_with_dipoles.png')


## Dipole group 3 (N400 response)
foci = []
for subject in subjects:
    dips = mne.read_dipole(f'{data_path}/dipoles/sub-{subject:02d}/meg/sub-{subject:02d}_task-epasana_dipoles_talairach.bdip')
    dip_selection = pd.read_csv(f'{data_path}/dipoles/sub-{subject:02d}/meg/sub-{subject:02d}_task-epasana_dipole_selection.tsv', sep='\t', index_col=0)
    pos_tal = dips.pos * 1000
    if 'LeftTemp3' in dip_selection.index:
        foci.append(pos_tal[dip_selection.loc['LeftTemp3'].dipole])
foci = np.array(foci)
brain = ga_stc.copy().crop(0.3, 0.5).mean().plot(
    'fsaverage',
    subjects_dir=data_path,
    hemi='lh',
    background='white',
    cortex='low_contrast',
    surface='inflated',
    clim=dict(kind='value', lims=(3.5, 3.8, 7)),
    brain_kwargs=dict(show_toolbar=False),
)

# Save images without dipoles
for view in views:
    brain.show_view(view)
    brain.save_image(f'figures/landmark3_{view}_no_dipoles.png')

# Save images with dipoles
brain.add_foci(foci, map_surface=map_surface, hemi='lh', scale_factor=0.2)
for view in views:
    brain.show_view(view)
    brain.save_image(f'figures/landmark3_{view}_with_dipoles.png')
