"""
Plot the grand-average timecourses for each of the three dipole groups as
defined in Vartiainen et al. 2011.
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os

# Path where you downloaded the OSF data to: https://osf.io/nu2ep/
data_path = './data'

# Whether to overwrite the existing figure
overwrite = False

stimuli = pd.read_csv(f'{data_path}/stimuli.csv')
stimuli['type'] = stimuli['type'].astype('category')
timecourses = np.load(f'{data_path}/dipoles/timecourses.npz')

stimulus_types = ['noisy word', 'consonants', 'pseudoword', 'word', 'symbols']
colors = ['C0', 'C1', 'C2', 'C3', 'C4']

##
# Plot the dipole timecourses for three dipole groups
groups = ['LeftOcci1', 'LeftOcciTemp2', 'LeftTemp3']
group_desc = ['Type I', 'Type II', 'N400m']
intervals = [(0.070, 0.130), (0.115, 0.200), (0.300, 0.400)]

fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(8, 2))
for group, desc, interval, ax in zip(groups, group_desc, intervals, axes.flat):
    for stimulus_type, color in zip(stimulus_types, colors):
        tc = timecourses[group][stimuli.query(f'type == "{stimulus_type}"').index].mean(axis=0)
        ax.plot(timecourses['times'] * 1000, tc * 1e9, color=color, label=stimulus_type)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.axvspan(interval[0] * 1000, interval[1] * 1000, color='black', alpha=0.05, zorder=-1)
        ax.set_xlim(-100, 600)
        ax.set_xticks([-100, 0, 100, 200, 300, 400, 500])
        ax.set_yticks([-5, 5, 10, 15])
        ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_position(('data', 0))
        ax.set_title(desc)
        if group == groups[0]:
            ax.set_ylabel('Activity (nAm)')
        if group == groups[1]:
            ax.set_xlabel('Time (ms)')
plt.tight_layout()

if not os.path.exists('figures/dipole_timecourses.pdf') or overwrite:
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/dipole_timecourses.pdf')
