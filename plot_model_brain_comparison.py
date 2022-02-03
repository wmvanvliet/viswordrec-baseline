"""
Create the plots in which the behavior of the layers in the model are compared
to the behavior of the epasana dipoles.
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from scipy.stats import zscore, rankdata
import pickle
import os

# Path where you downloaded the OSF data to: https://osf.io/nu2ep/
data_path = './data'

# Whether to overwrite the existing figure and data
overwrite = False

# Brain landmarks to show
selected_landmarks = ['LeftOcci1', 'LeftOcciTemp2', 'LeftTemp3']

# Stimulus types in the order in which they should be plotted
stimulus_types = ['word', 'pseudoword', 'consonants', 'symbols', 'noisy word']

# The subjects to plot the data for
subjects = range(1, 16)

# Read in the metadata and dipole information from each subject
metadata = []
dip_timecourses = []
dip_selection = []
for subject in tqdm(subjects):
    # Metadata: what stimulus was shown during each epoch?
    m = pd.read_csv(f'{data_path}/events/sub-{subject:02d}_task-epasana_events.tsv', sep='\t')
    m['subject'] = subject
    # Select only relevant trials
    m = m[m.value >= 2]
    m = m[m.value <= 11]
    metadata.append(m)

    # Dipole timecourses for each epoch (make sure it's in the correct order!)
    data = np.load(f'{data_path}/dipoles/sub-{subject:02d}/meg/sub-{subject:02d}_task-epasana_dipole_timecourses.npz')
    dip_t = data['proj']
    dip_t = dip_t[:, np.argsort(m.tif_file), :]
    dip_timecourses.append(dip_t)

    # Select the dipoles corresponding to the landmarks.
    # These are different for each subject.
    dip_sel = pd.read_csv(f'{data_path}/dipoles/sub-{subject:02d}/meg/sub-{subject:02d}_task-epasana_dipole_selection.tsv', sep='\t')
    dip_sel['subject'] = subject
    dip_selection.append(dip_sel)
metadata = pd.concat(metadata, ignore_index=True)
dip_selection = pd.concat(dip_selection, ignore_index=True)

## For each landmark, we analyze a different time interval

# Time ranges used for statistical analysis in Vartiainen et al. 2011
times = (np.arange(dip_t.shape[2]) - 140) / 600
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

## For each subject, compute mean dipole activation for each landmark
landmark_acts = []
for subject, sub_dip_timecourses in zip(subjects, dip_timecourses):
    dip_info = dip_selection.query(f'subject=={subject}')
    landmark_act = metadata.query(f'subject=={subject}').sort_values('tif_file')
    for dip, landmark, neg in zip(dip_info.dipole, dip_info.group, dip_info.neg):
        tc = sub_dip_timecourses[dip] * (-1 if neg else 1)
        quant = zscore(tc[:, time_rois[landmark]].mean(axis=1))
        landmark_act[landmark] = quant
    landmark_acts.append(landmark_act)
landmark_acts = pd.concat(landmark_acts, ignore_index=True)

## For three landmarks, show the mean dipole activation for each stimulus.
mean_acts = []
fig, axes = plt.subplots(1, len(selected_landmarks), sharex=True, sharey=True, figsize=(4.4, 3))
for landmark, ax in zip(selected_landmarks, axes.flat):
    x_offset = 0
    mean_type_acts = []
    for i, stimulus_type in enumerate(stimulus_types):
        mean_act = landmark_acts.query(f'type=="{stimulus_type}"').groupby('tif_file').agg('mean')[landmark]
        mean_type_acts.append(mean_act)
        ax.scatter(np.arange(len(mean_act)) + x_offset, mean_act, s=1, alpha=0.2)
        ax.plot([x_offset, x_offset + len(mean_act) - 1], np.repeat(mean_act.mean(), 2), color=plt.get_cmap('tab10').colors[i], label=stimulus_type)
        ax.xaxis.set_visible(False)
        ax.set_facecolor('#eee')
        ax.set_facecolor('#fafbfc')
        for pos in ['top', 'bottom', 'left', 'right']:
            ax.spines[pos].set_visible(False)
        x_offset += len(mean_act)
    mean_acts.append(np.hstack(mean_type_acts))
    if landmark == selected_landmarks[0]:
        ax.legend()
        ax.set_ylabel('Activation (z-scores)')
        #ax.set_ylabel('Activation (fT)')
    else:
        ax.yaxis.set_visible(False)
    if landmark == 'LeftOcci1':
        ax.set_title('Left Occi.\n65-115 ms', fontsize=10)
        ax.spines['left'].set_visible(True)
    elif landmark == 'LeftOcciTemp2':
        ax.set_title('Left Occi.Temp.\n140-200 ms', fontsize=10)
    elif landmark == 'LeftTemp3':
        ax.set_title('Left Temp.\n300-500 ms', fontsize=10)
    else:
        ax.set_title(landmark, fontsize=10)
plt.tight_layout()
mean_acts = np.array(mean_acts)

if not os.path.exists('figures/dipole_activation_patterns.pdf') or overwrite:
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/dipole_activation_patterns.pdf')

## Load model layer activations
with open(f'{data_path}/model_layer_activity.pkl', 'rb') as f:
    d = pickle.load(f)

layer_activity = zscore(d['mean_activity'], axis=1)
layer_names = d['layer_names']
layer_acts = pd.DataFrame(layer_activity.T, columns=layer_names)

## Show the behavior of each model layer for each stimulus
stimuli = pd.read_csv(f'{data_path}/stimuli.csv')
stimuli['type'] = stimuli['type'].astype('category')
fig, axes = plt.subplots(1, len(layer_activity), sharex=True, sharey=True, figsize=(10, 3))
for i, (act, ax) in enumerate(zip(layer_activity, axes.flat)):
    x_offset = 0
    for j, cat in enumerate(stimulus_types):
        cat_index = np.flatnonzero(stimuli['type'] == cat)
        selection = act[cat_index]
        ax.scatter(np.arange(len(selection)) + x_offset, selection, s=1, alpha=0.2)
        ax.plot([x_offset, x_offset+len(selection) - 1], np.repeat(selection.mean(), 2), color=plt.get_cmap('tab10').colors[j], label=cat)
        ax.xaxis.set_visible(False)
        ax.set_facecolor('#eee')
        ax.set_facecolor('#fafbfc')
        ax.set_title(f'{layer_names[i]}\n', fontsize=10)
        for pos in ['top', 'bottom', 'left', 'right']:
            ax.spines[pos].set_visible(False)
        if i == 0:
            ax.legend()
            ax.set_ylabel('Activation (z-scores)')
            ax.spines['left'].set_visible(True)
        else:
            ax.yaxis.set_visible(False)
        x_offset += len(selection)
plt.tight_layout()

if not os.path.exists('figures/model_activation_patterns.pdf') or overwrite:
    os.makedirs('figures', exist_ok=True)
    plt.savefig('figures/model_activation_patterns.pdf')

## Save giant CSV file that will be used for single-subject statistics later.
data = landmark_acts.join(stimuli[['tif_file']].join(layer_acts).set_index('tif_file'), on='tif_file')
if not os.path.exists(f'{data_path}/brain_model_comparison/brain_model_comparison_dipoles.csv') or overwrite:
    os.makedirs(f'{data_path}/brain_model_comparison', exist_ok=True)
    data.to_csv(f'{data_path}/brain_model_comparison/brain_model_comparison_dipoles.csv')

# Compute some basic statistics
ga_data = data.groupby('tif_file').mean()
print(ga_data.corr().loc['conv1_relu':'word_relu', selected_landmarks])

#             LeftOcci1  LeftOcciTemp2  LeftTemp3
# conv1_relu   0.792663      -0.476589  -0.348777
# conv2_relu   0.791467      -0.475489  -0.347582
# conv3_relu   0.763833      -0.457302  -0.337939
# conv4_relu   0.601714      -0.302675  -0.204164
# conv5_relu   0.687968      -0.337723  -0.209251
# fc1_relu    -0.321628       0.385759   0.392012
# fc2_relu    -0.526866       0.466570   0.434970
# word_relu   -0.615212       0.455502   0.440379
