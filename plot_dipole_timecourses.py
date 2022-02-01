"""
Compare grand-average activity at the Epasana dipoles with the model activity.
"""
import mne
import numpy as np
from tqdm import tqdm
import pandas as pd
from matplotlib import pyplot as plt

from config import fname, subjects

data_path = './data'

metadata = []
dip_timecourses = []
dip_selection = []
for subject in tqdm(subjects):
    m = pd.read_csv(f'{data_path}/events/sub-{subject:02d}_task-epasana_events.tsv', sep='\t')
    data = np.load(f'{data_path}/dipoles/sub-{subject:02d}/meg/sub-{subject:02d}_task-epasana_dipole_timecourses.npz')
    dip_t = data['proj']
    times = data['times']
    dip_t = dip_t[:, np.argsort(m.tif_file), :]
    dip_timecourses.append(dip_t)
    dip_sel = pd.read_csv(fname.dip_selection(subject=subject), sep='\t')
    dip_sel['subject'] = subject
    dip_selection.append(dip_sel)
    m['subject'] = subject
    metadata.append(m)
metadata = pd.concat(metadata, ignore_index=True)
dip_selection = pd.concat(dip_selection, ignore_index=True)

stimuli = pd.read_csv(fname.stimulus_selection)
stimuli['type'] = stimuli['type'].astype('category')

##
# Time ranges used for statistical analysis in Vartiainen et al. 2011
time_rois = {
    'LeftOcci1': slice(*np.searchsorted(times, [0.065, 0.115])),
    'RightOcci1': slice(*np.searchsorted(times, [0.065, 0.115])),
    'LeftOcciTemp2': slice(*np.searchsorted(times, [0.14, 0.2])),
    'RightOcciTemp2': slice(*np.searchsorted(times, [0.185, 0.220])),
    'LeftTemp3': slice(*np.searchsorted(times, [0.300, 0.400])),
    'RightTemp3': slice(*np.searchsorted(times, [0.300, 0.400])),
    'LeftFront2-3': slice(*np.searchsorted(times, [0.300, 0.500])),
    'RightFront2-3': slice(*np.searchsorted(times, [0.300, 0.500])),
    'LeftPar2-3': slice(*np.searchsorted(times, [0.250, 0.350])),
}

groups = ['LeftOcci1', 'LeftOcciTemp2', 'LeftTemp3']
group_desc = ['Occipital', 'Occipital-temporal', 'Temporal']
intervals = [(0.064, 0.115), (0.114, 0.200), (0.300, 0.500)]
fig, axes = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(8, 2))
for group, desc, interval, ax in zip(groups, group_desc, intervals, axes.flat):
    group_meta = []
    group_tcs = []
    sel = dip_selection.query(f'group=="{group}" and subject==14')
    for sub, dip, neg in zip(sel.subject, sel.dipole, sel.neg):
        tc = dip_timecourses[sub - 1][dip] * (-1 if neg else 1) * 1E9
        group_tcs.append(tc)
        df = metadata.query(f'subject=={sub}').sort_values('tif_file')
        group_meta.append(df)
    group_meta = pd.concat(group_meta, ignore_index=True)
    group_tcs = np.vstack(group_tcs)

    stimulus_types = ['noisy word', 'consonants', 'pseudoword', 'word', 'symbols']
    colors = ['C0', 'C1', 'C2', 'C3', 'C4']
    for stimulus_type, color in zip(stimulus_types, colors):
        tc = group_tcs[group_meta.query(f'type == "{stimulus_type}"').index].mean(axis=0)
        ax.plot(times * 1000, tc, color=color, label=stimulus_type)
        ax.axhline(0, color='black', linewidth=0.5)
        ax.set_xlim(-100, 600)
        ax.set_xticks([-100, 0, 100, 200, 300, 400, 500])
        ax.set_yticks([-5, 5, 10, 15])
        #ax.spines['bottom'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_position(('data', 0))
        ax.set_title(desc)
        if group == groups[0]:
            ax.set_ylabel('Activity (nAm)')
        if group == groups[1]:
            ax.set_xlabel('Time (ms)')
plt.tight_layout()
plt.savefig('figures/dipole_timecourses.pdf')
