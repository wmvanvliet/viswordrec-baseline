import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.linear_model import LinearRegression

# Path where you downloaded the OSF data to: https://osf.io/nu2ep/
data_path = './data'

# Model you want to correlate with the brain data
model_name = "vgg11stochastic_first_imagenet_then_10kwords-freq"

layer_names = [
    "conv1_relu",
    "conv2_relu",
    "conv3_relu",
    "conv4_relu",
    "conv5_relu",
    "fc1_relu",
    "fc2_relu",
    "word_relu",
]
landmarks = ["LeftOcci1", "LeftOcciTemp2", "LeftTemp3"]
stimulus_types = ["noisy word", "symbols", "consonants", "word", "pseudoword"]
colors = ["#1f77b4", "#9467bd", "#ff7f0e", "#d62728", "#2ca02c"]

stimuli = pd.read_csv(f'{data_path}/stimuli.csv')
stimuli['type'] = stimuli['type'].astype('category')

with open(f'{data_path}/model_layer_activity/{model_name}_mean_layer_activity_torch19.pkl', 'rb') as f:
    d = pickle.load(f)

layer_activity = zscore(d['mean_activity'], axis=1)
layer_acts = pd.DataFrame(layer_activity.T, columns=d['layer_names'])
dipole_acts = pd.read_csv(f'{data_path}/dipoles/grand_average_dipole_activation.csv')
data = dipole_acts.join(layer_acts)

def plt_cmp(landmark, layer):
    fig = plt.figure(figsize=(4, 4))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=(6, 1),
        height_ratios=(1, 6),
        left=0.2,
        right=0.95,
        bottom=0.15,
        top=0.95,
        wspace=0.03,
        hspace=0.03,
    )
    ax = fig.add_subplot(gs[1, 0])
    ax_distx = fig.add_subplot(gs[0, 0], sharex=ax)
    ax_disty = fig.add_subplot(gs[1, 1], sharey=ax)
    ax_distx.xaxis.set_visible(False)
    ax_distx.yaxis.set_visible(False)
    ax_disty.xaxis.set_visible(False)
    ax_disty.yaxis.set_visible(False)
    for stimulus_type, color in zip(stimulus_types, colors):
        data_sel = data.query(f'type == "{stimulus_type}"')
        ax.scatter(
            data_sel[landmark],
            data_sel[layer],
            s=5,
            alpha=0.3,
            color=color,
            label=stimulus_type,
        )
        sns.kdeplot(data=data_sel, x=landmark, ax=ax_distx, color=color)
        sns.kdeplot(data=data_sel, y=layer, ax=ax_disty, color=color)
    ax.set_xlabel(f"{landmark} (z-scores)")
    ax.set_ylabel(f"{layer} (z-scores)")

    xmin = data[landmark].min()
    xmax = data[landmark].max()
    ymin = data[layer].min()
    ymax = data[layer].max()
    ax.set_xlim(xmin - 0.1, xmax + 0.1)
    ax.set_ylim(ymin - 0.5, ymax + 0.5)

    pmin, pmax = (
        LinearRegression()
        .fit(data[[landmark]].values, data[layer].values)
        .predict([[xmin], [xmax]])
    )
    ax.plot([xmin, xmax], [pmin, pmax], color="black", linewidth=1, zorder=-100)
    r = data.corr(numeric_only=True).at[landmark, layer]
    rstr = f"{r:.2f}"[1:]
    ax.annotate(
        f"r={rstr}",
        (xmax, pmax),
        xytext=(-2, 0),
        textcoords="offset points",
        horizontalalignment="right",
    )


plt_cmp("LeftOcci1", "conv1_relu")
plt.savefig(f"figures/{model_name}_LeftOcci1_vs_conv1_relu.pdf")
plt_cmp("LeftOcciTemp2", "fc1_relu")
plt.savefig(f"figures/{model_name}_LeftOcciTemp2_vs_fc1_relu.pdf")
plt_cmp("LeftTemp3", "word_relu")
plt.savefig(f"figures/{model_name}_LeftTemp3_vs_word_relu.pdf")
