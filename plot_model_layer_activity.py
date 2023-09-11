"""
Create the plots of the behavior of the layers in the model.
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import zscore, ttest_ind
from statsmodels.stats.multitest import multipletests

# Path where you downloaded the OSF data to: https://osf.io/nu2ep/
data_path = './data'

# Stimulus types in the order in which they should be plotted
stimulus_types = ['noisy word', 'symbols', 'consonants', 'word', 'pseudoword']
colors = ['#1f77b4', '#9467bd', '#ff7f0e', '#d62728', '#2ca02c']
contrasts = [('noisy word', 'symbols'),
             ('symbols', 'consonants'),
             ('consonants', 'word'),
             ('word', 'pseudoword')]

stimuli = pd.read_csv(f'{data_path}/stimuli.csv')
stimuli['type'] = stimuli['type'].astype('category')

def effect_size(x, y):
    pvalue = ttest_ind(x, y).pvalue
    dvalue = np.mean(x) - np.mean(y)
    return dvalue, pvalue


## Load model layer activations
import pickle

# Main contrasts in the paper
model_names = [
    'vgg11_imagenet',
    'vgg11_first_imagenet_then_250words',
    'vgg11stochastic_first_imagenet_then_250words',
    'vgg11stochastic_first_imagenet_then_1kwords',
    'vgg11stochastic_first_imagenet_then_10kwords',
    'vgg11stochastic_first_imagenet_then_10kwords-freq',
]

# # Exploring the effect of noise
# model_names = [
#     'vgg11nobn_first_imagenet_then_250words',
#     'vgg11_first_imagenet_then_250words',
#     'vgg11stochasticnobn_first_imagenet_then_250words',
#     'vgg11stochastic_first_imagenet_then_250words',
# ]

# # Exploring model stability
# model_names = [
#     'vgg11stochastic_first_imagenet_then_10kwords-freq_iter-1',
#     'vgg11stochastic_first_imagenet_then_10kwords-freq_iter-2',
#     'vgg11stochastic_first_imagenet_then_10kwords-freq_iter-3',
#     'vgg11stochastic_first_imagenet_then_10kwords-freq_iter-4',
#     'vgg11stochastic_first_imagenet_then_10kwords-freq_iter-5',
#     'vgg11stochastic_first_imagenet_then_10kwords-freq_iter-6',
#     'vgg11stochastic_first_imagenet_then_10kwords-freq_iter-7',
#     'vgg11stochastic_first_imagenet_then_10kwords-freq_iter-8',
#     'vgg11stochastic_first_imagenet_then_10kwords-freq_iter-9',
#     'vgg11stochastic_first_imagenet_then_10kwords-freq_iter-10',
# ]

# # Exploring the effect of architecture
# model_names = [
#       'vgg11c3stochastic_10kwords-freq',
#       'vgg11c4stochastic_10kwords-freq',
#       'vgg11c6stochastic_10kwords-freq',
#       'vgg11l1stochastic_first_imagenet_then_10kwords-freq',
#       'vgg11l2stochastic_first_imagenet_then_10kwords-freq',
#       'vgg11stochastic_first_imagenet_then_10kwords-freq',
#       'vgg11l4stochastic_first_imagenet_then_10kwords-freq',
# ]

# # Exploring the effect of imagenet pretraining
# model_names = [
#     'vgg11stochastic_random',
#     'vgg11stochastic_imagenet',
#     'vgg11stochastic_10kwords-freq',
#     'vgg11stochastic_first_imagenet_then_10kwords-freq',
# ]

# # Exploring the effect of classifier width
# model_names = [
#     'vgg11stochastic-cs1024_first_imagenet_then_10kwords-freq',
#     'vgg11stochastic-cs2048_first_imagenet_then_10kwords-freq',
#     'vgg11stochastic_first_imagenet_then_10kwords-freq',
#     'vgg11stochastic-cs8192_first_imagenet_then_10kwords-freq',
# ]

layer_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc1', 'fc2', 'word']

model_layer_activity = []
model_effect_sizes = []
fig, axes = plt.subplots(len(model_names), len(layer_names), sharex=True, sharey=True, figsize=(8, 2 * len(model_names)), squeeze=False)
for i, model_name in enumerate(model_names):
    with open(f'{data_path}/model_layer_activity/{model_name}_mean_layer_activity_torch19.pkl', 'rb') as f:
        d = pickle.load(f)

    layer_activity = zscore(d['mean_activity'], axis=1)
    layer_acts = pd.DataFrame(layer_activity.T, columns=d['layer_names'])

    effect_sizes = []

    # Show the behavior of each model layer for each stimulus
    for j, act in enumerate(layer_activity):
        ax = axes[i, j]
        layer_effect_sizes = []
        for k, cat in enumerate(stimulus_types):
            cat_index = np.flatnonzero(stimuli['type'] == cat)
            selection = act[cat_index]
            mean = np.mean(act[cat_index])
            v = ax.violinplot(selection, [k/3], showmeans=False, showextrema=False)
            for b in v['bodies']:
                # get the center
                m = np.mean(b.get_paths()[0].vertices[:, 0])
                # modify the paths to not go further right than the center
                b.get_paths()[0].vertices[:, 0] = np.clip(b.get_paths()[0].vertices[:, 0], -np.inf, m)
                # set the color
                b.set_facecolor(colors[k])
            ax.plot([k/3-0.2, k/3-0.02], np.repeat(mean, 2), color=colors[k])
            ax.xaxis.set_visible(False)
            ax.set_facecolor('#eee')
            for pos in ['top', 'bottom', 'left', 'right']:
                ax.spines[pos].set_visible(False)
            if j == 0:
                ax.set_ylabel('Activation (z-scores)')
                ax.spines['left'].set_visible(True)
            else:
                ax.yaxis.set_visible(False)
        for a, b, in contrasts:
            vals_a = layer_activity[j, stimuli.query(f'type=="{a}"').y]
            vals_b = layer_activity[j, stimuli.query(f'type=="{b}"').y]
            layer_effect_sizes.append(effect_size(vals_a, vals_b))
        effect_sizes.append(layer_effect_sizes)

    model_layer_activity.append(layer_activity)
    model_effect_sizes.append(effect_sizes)

# Perform FDR correction on the p-values
is_significant = multipletests([p for a in model_effect_sizes for b in a for d, p in b], method='fdr_bh')[0]
is_significant_iter = iter(is_significant)

# Annotate the plot with significant differences
for i, model_name in enumerate(model_names):
    print('\n' + model_name)
    layer_activity = model_layer_activity[i]
    for j, act in enumerate(layer_activity):
        print('    ' + layer_names[j])
        ax = axes[i][j]
        layer_effect_sizes = model_effect_sizes[i][j]

        bar_x1 = -0.1
        bar_x2 = 0.2
        for (d, p), (a, b) in zip(layer_effect_sizes, contrasts):
            vals_a = layer_activity[j, stimuli.query(f'type=="{a}"').y]
            vals_b = layer_activity[j, stimuli.query(f'type=="{b}"').y]
            s = next(is_significant_iter)
            print(f'        {a} vs {b}: d={d} p={p} s={s}')
            if s:
                props = {'connectionstyle':'bar','arrowstyle':'-',\
                         'shrinkA':20,'shrinkB':20,'linewidth':0.5, 'color':'#666'}
                annoty = max(vals_a.max(), vals_b.max()) + 0.1
                if round(abs(d), 2) >= 1:
                    dannot = f'{abs(d):.1f}'
                else:
                    dannot = f'{abs(d):.2f}'.lstrip('0')
                ax.annotate(
                    f'{dannot}',
                    xy=((bar_x1 + bar_x2) / 2, annoty),
                    xytext=(0, 3), textcoords='offset points',
                    zorder=10,
                    horizontalalignment='center',
                    verticalalignment='bottom',
                    fontsize=7,
                    color='#666',
                )
                ax.annotate('', xy=(bar_x1, annoty), xytext=(bar_x2, annoty), arrowprops=props, color='#666')
            bar_x1 += 0.37
            bar_x2 += 0.37

plt.tight_layout(h_pad=0.4, w_pad=0.2, pad=0)
