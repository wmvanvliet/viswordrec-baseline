"""
Create the plots in which the behavior of the layers in the model are compared
to the behavior of the epasana dipoles.
"""

import os
import pickle

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator
from scipy.stats import ttest_ind, zscore
from statsmodels.stats.multitest import fdrcorrection

# Path to the OSF downloaded data
data_path = "./data"

# Whether to overwrite the existing figure
overwrite = False

# The filename for the produced figure
fig_fname = "figures/model_brain_comparison.pdf"

stimuli = pd.read_csv(f"{data_path}/stimuli.csv")
stimuli["type"] = stimuli["type"].astype("category")
dip_act = pd.read_csv(f"{data_path}/dipoles/grand_average_dipole_activation.csv")

# Brain landmarks to show
selected_landmarks = [
    "LeftOcci1",
    "LeftOcciTemp2",
    "LeftTemp3",
    "RightOcci1",
    "RightOcciTemp2",
    "RightTemp3",
]

# Stimulus types in the order in which they should be plotted
stimulus_types = ["noisy word", "symbols", "consonants", "word", "pseudoword"]
colors = ["#1f77b4", "#9467bd", "#ff7f0e", "#d62728", "#2ca02c"]


noise_ceiling = pd.read_csv(f"{data_path}/dipoles/noise_ceiling.csv", index_col=0)

## Load model layer activations

# For Figure 4 in the manuscript
model_names = [
    "vgg11_imagenet",
    "vgg11_first_imagenet_then_250words",
    "vgg11stochastic_first_imagenet_then_250words",
    "vgg11stochastic_first_imagenet_then_1kwords",
    "vgg11stochastic_first_imagenet_then_10kwords",
    "vgg11stochastic_first_imagenet_then_10kwords-freq",
]
annotations_titles = ["noisy\nactivations", "vocab\nsize", "frequency\nbalanced"]
model_annotations = [
    ["NO", "0", "NO"],
    ["NO", "250", "NO"],
    ["YES", "250", "NO"],
    ["YES", "1000", "NO"],
    ["YES", "10 000", "NO"],
    ["YES", "10 000", "YES"],
]
n_layers = 8
width_ratios = [2, n_layers, 1]
layer_landmark_correspondence = [
    dict(LeftOcci1=0, LeftOcciTemp2=None, LeftTemp3=None),
    dict(LeftOcci1=0, LeftOcciTemp2=None, LeftTemp3=None),
    dict(LeftOcci1=0, LeftOcciTemp2=5, LeftTemp3=None),
    dict(LeftOcci1=0, LeftOcciTemp2=5, LeftTemp3=7),
    dict(LeftOcci1=0, LeftOcciTemp2=5, LeftTemp3=None),
    dict(LeftOcci1=0, LeftOcciTemp2=5, LeftTemp3=7),
]

# # For Figure 5 in the manuscript
# model_names = [
#     "vgg11c4stochastic_10kwords-freq",
#     "vgg11l2stochastic_first_imagenet_then_10kwords-freq",
#     "vgg11c6stochastic_10kwords-freq",
#     "vgg11l4stochastic_first_imagenet_then_10kwords-freq",
#     "vgg11stochastic_first_imagenet_then_10kwords-freq",
# ]
# annotations_titles = ["conv.\nlayers", "f.c.\n layers"]
# model_annotations = [
#     ["4", "3"],
#     ["5", "2"],
#     ["6", "3"],
#     ["5", "4"],
#     ["5", "3"],
# ]
# layer_landmark_correspondence = [
#     dict(LeftOcci1=0, LeftOcciTemp2=None, LeftTemp3=None),
#     dict(LeftOcci1=0, LeftOcciTemp2=None, LeftTemp3=None),
#     dict(LeftOcci1=0, LeftOcciTemp2=None, LeftTemp3=6),
#     dict(LeftOcci1=0, LeftOcciTemp2=5, LeftTemp3=7),
#     dict(LeftOcci1=0, LeftOcciTemp2=5, LeftTemp3=7),
#     dict(LeftOcci1=0, LeftOcciTemp2=5, LeftTemp3=7),
# ]
# n_layers = 9
# width_ratios = [1.3, n_layers, 1]

# # Supplementary figure 2
# model_names = [
#     "vgg11nobn_first_imagenet_then_250words",
#     "vgg11_first_imagenet_then_250words",
#     "vgg11stochasticnobn_first_imagenet_then_250words",
#     "vgg11stochastic_first_imagenet_then_250words",
# ]
# annotations_titles = ["noisy\nactivations", "batch norm"]
# model_annotations = [
#     ["NO", "NO"],
#     ["NO", "YES"],
#     ["YES", "NO"],
#     ["YES", "YES"],
# ]
# n_layers = 8
# width_ratios = [1.5, n_layers, 1]
# layer_landmark_correspondence = [
#     dict(LeftOcci1=0, LeftOcciTemp2=None, LeftTemp3=None),
#     dict(LeftOcci1=0, LeftOcciTemp2=None, LeftTemp3=None),
#     dict(LeftOcci1=0, LeftOcciTemp2=None, LeftTemp3=None),
#     dict(LeftOcci1=0, LeftOcciTemp2=4, LeftTemp3=None),
# ]

# # Supplementary figure 3
# model_names = [
#     "vgg11stochastic_first_imagenet_then_10kwords-freq_iter-1",
#     "vgg11stochastic_first_imagenet_then_10kwords-freq_iter-2",
#     "vgg11stochastic_first_imagenet_then_10kwords-freq_iter-3",
#     "vgg11stochastic_first_imagenet_then_10kwords-freq_iter-4",
#     "vgg11stochastic_first_imagenet_then_10kwords-freq_iter-5",
#     "vgg11stochastic_first_imagenet_then_10kwords-freq_iter-6",
#     "vgg11stochastic_first_imagenet_then_10kwords-freq_iter-7",
#     "vgg11stochastic_first_imagenet_then_10kwords-freq_iter-8",
#     "vgg11stochastic_first_imagenet_then_10kwords-freq_iter-9",
#     "vgg11stochastic_first_imagenet_then_10kwords-freq_iter-10",
# ]
# annotations_titles = ["iteration"]
# model_annotations = [
#     ["1"],
#     ["2"],
#     ["3"],
#     ["4"],
#     ["5"],
#     ["6"],
#     ["7"],
#     ["8"],
#     ["9"],
#     ["10"],
# ]
# layer_landmark_correspondence = [
#     dict(LeftOcci1=0, LeftOcciTemp2=5, LeftTemp3=7),
#     dict(LeftOcci1=0, LeftOcciTemp2=5, LeftTemp3=7),
#     dict(LeftOcci1=0, LeftOcciTemp2=5, LeftTemp3=7),
#     dict(LeftOcci1=0, LeftOcciTemp2=5, LeftTemp3=7),
#     dict(LeftOcci1=0, LeftOcciTemp2=5, LeftTemp3=7),
#     dict(LeftOcci1=0, LeftOcciTemp2=5, LeftTemp3=7),
#     dict(LeftOcci1=0, LeftOcciTemp2=5, LeftTemp3=7),
#     dict(LeftOcci1=0, LeftOcciTemp2=5, LeftTemp3=7),
#     dict(LeftOcci1=0, LeftOcciTemp2=5, LeftTemp3=7),
#     dict(LeftOcci1=0, LeftOcciTemp2=5, LeftTemp3=7),
# ]
# n_layers = 8
# width_ratios = [0.75, n_layers, 1]

# # Supplementary figure 4
# model_names = [
#     "vgg11stochastic-cs1024_first_imagenet_then_10kwords-freq",
#     "vgg11stochastic-cs2048_first_imagenet_then_10kwords-freq",
#     "vgg11stochastic_first_imagenet_then_10kwords-freq",
#     "vgg11stochastic-cs8192_first_imagenet_then_10kwords-freq",
# ]
# annotations_titles = ["f.c. layer width"]
# model_annotations = [
#     ["1024"],
#     ["2048"],
#     ["4096"],
#     ["8192"],
# ]
# layer_landmark_correspondence = [
#     dict(LeftOcci1=0, LeftOcciTemp2=6, LeftTemp3=7),
#     dict(LeftOcci1=0, LeftOcciTemp2=5, LeftTemp3=6),
#     dict(LeftOcci1=0, LeftOcciTemp2=5, LeftTemp3=7),
#     dict(LeftOcci1=0, LeftOcciTemp2=5, LeftTemp3=6),
# ]
# n_layers = 8
# width_ratios = [1, n_layers, 1]

# # Supplementary figure 5
# model_names = [
#     "vgg11stochastic-nl0.05_first_imagenet_then_10kwords-freq",
#     "vgg11stochastic_first_imagenet_then_10kwords-freq",
#     "vgg11stochastic-nl0.2_first_imagenet_then_10kwords-freq",
#     "vgg11stochastic-nl0.3_first_imagenet_then_10kwords-freq",
#     "vgg11stochastic-nl0.4_first_imagenet_then_10kwords-freq",
#     "vgg11stochastic-nl0.5_first_imagenet_then_10kwords-freq",
# ]
# annotations_titles = ["noise level"]
# model_annotations = [
#     ["0.05"],
#     ["0.1"],
#     ["0.2"],
#     ["0.3"],
#     ["0.4"],
#     ["0.5"],
# ]
# layer_landmark_correspondence = [
#     dict(LeftOcci1=0, LeftOcciTemp2=5, LeftTemp3=6),
#     dict(LeftOcci1=0, LeftOcciTemp2=5, LeftTemp3=7),
#     dict(LeftOcci1=0, LeftOcciTemp2=5, LeftTemp3=7),
#     dict(LeftOcci1=0, LeftOcciTemp2=5, LeftTemp3=7),
#     dict(LeftOcci1=0, LeftOcciTemp2=5, LeftTemp3=7),
#     dict(LeftOcci1=0, LeftOcciTemp2=5, LeftTemp3=7),
# ]
# n_layers = 8
# width_ratios = [1, n_layers, 1]

# # Supplementary figure 6
# model_names = [
#     "vgg11stochastic_first_imagenet_then_250words",
#     "vgg11stochastic_first_imagenet_then_1kwords",
#     "vgg11stochastic_first_imagenet_then_5kwords",
#     "vgg11stochastic_first_imagenet_then_10kwords",
#     "vgg11stochastic_first_imagenet_then_10kwords-freq",
# ]
# annotations_titles = ["vocab\nsize", "frequency\nbalanced"]
# model_annotations = [
#     ["250", "NO"],
#     ["1000", "NO"],
#     ["5000", "NO"],
#     ["10 000", "NO"],
#     ["10 000", "YES"],
# ]
# n_layers = 8
# width_ratios = [1.75, n_layers, 1]
# layer_landmark_correspondence = [
#     dict(LeftOcci1=0, LeftOcciTemp2=5, LeftTemp3=None),
#     dict(LeftOcci1=0, LeftOcciTemp2=5, LeftTemp3=7),
#     dict(LeftOcci1=0, LeftOcciTemp2=5, LeftTemp3=7),
#     dict(LeftOcci1=0, LeftOcciTemp2=5, LeftTemp3=None),
#     dict(LeftOcci1=0, LeftOcciTemp2=5, LeftTemp3=7),
# ]

# # Supplementary figure 7
# model_names = [
#     "vgg11stochastic_first_imagenet_then_10kwords-freq-0.01",
#     "vgg11stochastic_first_imagenet_then_10kwords-freq-0.1",
#     "vgg11stochastic_first_imagenet_then_10kwords-freq",
#     "vgg11stochastic_first_imagenet_then_10kwords-freq-0.3",
#     "vgg11stochastic_first_imagenet_then_10kwords-freq-0.4",
#     "vgg11stochastic_first_imagenet_then_10kwords-freq-0.5",
# ]
# annotations_titles = ["frequency\nbalancing"]
# model_annotations = [
#     ["0.01"],
#     ["0.1"],
#     ["0.2"],
#     ["0.3"],
#     ["0.4"],
#     ["0.5"],
# ]
# n_layers = 8
# width_ratios = [1, n_layers, 1]
# layer_landmark_correspondence = [
#     dict(LeftOcci1=0, LeftOcciTemp2=5, LeftTemp3=7),
#     dict(LeftOcci1=0, LeftOcciTemp2=5, LeftTemp3=7),
#     dict(LeftOcci1=0, LeftOcciTemp2=5, LeftTemp3=7),
#     dict(LeftOcci1=0, LeftOcciTemp2=None, LeftTemp3=6),
#     dict(LeftOcci1=0, LeftOcciTemp2=5, LeftTemp3=7),
#     dict(LeftOcci1=0, LeftOcciTemp2=5, LeftTemp3=7),
# ]

# # Supplementary figure 8
# model_names = [
#     "vgg11stochastic_random",
#     "vgg11stochastic_imagenet",
#     "vgg11stochastic_10kwords-freq",
#     "vgg11stochastic_first_imagenet_then_10kwords-freq",
# ]
# annotations_titles = ["training regime"]
# model_annotations = [
#     ["Random weights"],
#     ["Only ImageNet"],
#     ["Only words"],
#     ["Fist ImageNet,\nthen words"],
# ]
# layer_landmark_correspondence = [
#     dict(LeftOcci1=0, LeftOcciTemp2=None, LeftTemp3=None),
#     dict(LeftOcci1=0, LeftOcciTemp2=None, LeftTemp3=None),
#     dict(LeftOcci1=0, LeftOcciTemp2=5, LeftTemp3=7),
#     dict(LeftOcci1=0, LeftOcciTemp2=5, LeftTemp3=7),
# ]
# n_layers = 8
# width_ratios = [2, n_layers, 1]

fig = plt.figure(figsize=(n_layers + 3, 2 * len(model_names)))
# fig = plt.figure(figsize=(7, 2 * len(model_names)))
gs_main = fig.add_gridspec(1, 3, width_ratios=width_ratios, wspace=0.19)
gs_annotations = gs_main[0].subgridspec(len(model_names), 1)
gs_response_patterns = gs_main[1].subgridspec(len(model_names), n_layers)
gs_correlations = gs_main[2].subgridspec(len(model_names), 1)

all_pvalues = []
all_ymax = []
all_effect_size = []
axes_response_patterns = []
axes_correlations = []
for i, model_name in enumerate(model_names):
    model_pvalues = []
    model_ymax = []
    model_effect_size = []
    model_correlations = []
    model_axes = []
    with open(
        f"{data_path}/model_layer_activity/{model_name}_mean_layer_activity_torch19.pkl",
        "rb",
    ) as f:
        d = pickle.load(f)

    layer_activity = zscore(d["mean_activity"], axis=1)
    layer_names = d["layer_names"]
    layer_acts = pd.DataFrame(layer_activity.T, columns=layer_names)

    ax_annotation = fig.add_subplot(gs_annotations[i])
    n_annotations = len(annotations_titles)
    step = 1 / (n_annotations + 1)
    if i == 0:
        for j, t in enumerate(annotations_titles, 1):
            ax_annotation.text(
                j * step, 1, t, va="top", ha="center", fontsize=12, rotation=90
            )
        for j, a in enumerate(model_annotations[i], 1):
            ax_annotation.text(j * step, 0.3, a, va="center", ha="center", fontsize=12)
    else:
        for j, a in enumerate(model_annotations[i], 1):
            ax_annotation.text(j * step, 0.5, a, va="center", ha="center", fontsize=12)
    ax_annotation.set_xlim(0, 0.8)
    ax_annotation.set_ylim(0, 1)
    ax_annotation.set_axis_off()

    # Show the behavior of each model layer for each stimulus
    for j, act in enumerate(layer_activity):
        layer_pvalues = []
        layer_ymax = []
        layer_effect_size = []
        layer_correlations = []
        ax = fig.add_subplot(gs_response_patterns[i, j])
        model_axes.append(ax)
        prev_selection = None
        for k, cat in enumerate(stimulus_types):
            cat_index = np.flatnonzero(stimuli["type"] == cat)
            selection = act[cat_index]
            mean = selection.mean()

            v = ax.violinplot(selection, [k / 3], showmeans=False, showextrema=False)
            for b in v["bodies"]:
                # get the center
                m = np.mean(b.get_paths()[0].vertices[:, 0])
                # modify the paths to not go further right than the center
                b.get_paths()[0].vertices[:, 0] = np.clip(
                    b.get_paths()[0].vertices[:, 0], -np.inf, m
                )
                # set the color
                b.set_facecolor(colors[k])
            ax.plot([k / 3 - 0.2, k / 3 - 0.02], np.repeat(mean, 2), color=colors[k])

            ax.set_ylim(-2.9, 2.9)
            ax.xaxis.set_visible(False)
            ax.set_facecolor("#fafbfc")
            for pos in ["top", "bottom", "left", "right"]:
                ax.spines[pos].set_visible(False)
            if j == 0:
                ax.set_ylabel("Activation (z-scores)")
                ax.yaxis.set_label_coords(-0.35, 0.5)
                ax.spines["left"].set_visible(True)
            else:
                ax.yaxis.set_visible(False)

            for k, l in enumerate(layer_landmark_correspondence[i].values()):
                if l == j:
                    ax.spines["left"].set_visible(True)
                    ax.spines["left"].set_color(f"C{k+6}")
                    ax.spines["right"].set_visible(True)
                    ax.spines["right"].set_color(f"C{k+6}")
                    ax.spines["top"].set_visible(True)
                    ax.spines["top"].set_color(f"C{k+6}")
                    ax.spines["bottom"].set_visible(True)
                    ax.spines["bottom"].set_color(f"C{k+6}")
            if i == len(model_names) - 1:
                ax.xaxis.set_visible(True)
                ax.xaxis.set_ticks([])
                ax.set_xlabel(f"{layer_names[j].split('_relu')[0]}\n", fontsize=10)
                ax.xaxis.set_label_coords(0.5, -0.05)

            if prev_selection is not None:
                t, p = ttest_ind(prev_selection, selection)
                layer_pvalues.append(p)
                layer_ymax.append(
                    max(
                        np.mean(prev_selection) + 2.5 * np.std(prev_selection),
                        np.mean(selection) + 2.5 * np.std(selection),
                    )
                )
                layer_effect_size.append(
                    abs(np.mean(prev_selection) - np.mean(selection))
                )
            prev_selection = selection
        model_pvalues.append(layer_pvalues)
        model_ymax.append(layer_ymax)
        model_effect_size.append(layer_effect_size)
    all_pvalues.append(model_pvalues)
    all_ymax.append(model_ymax)
    all_effect_size.append(model_effect_size)
    axes_response_patterns.append(model_axes)
    data = dip_act.join(
        stimuli[["tif_file"]].join(layer_acts).set_index("tif_file"), on="tif_file"
    )

    ax_correlations = fig.add_subplot(gs_correlations[i])
    axes_correlations.append(ax_correlations)
    for j, landmark in enumerate(selected_landmarks[:3]):
        ceiling = noise_ceiling.at[landmark, "ga_ceiling"]
        ax_correlations.plot(
            [-1, n_layers + 1],
            [ceiling, ceiling],
            color=f"C{j+6}",
            alpha=0.5,
            linewidth=2,
            zorder=0,
        )
        r = (
            data.groupby("tif_file")
            .mean(numeric_only=True)
            .corr()
            .loc[landmark, layer_names]
            .values
        )
        ax_correlations.plot(r, color=f"C{j+6}", marker="o", markersize=3)
        l = layer_landmark_correspondence[i][landmark]
        if l is not None:
            ax_correlations.scatter([l], [r[l]], marker="s", color=f"C{j+6}", s=40)
    for pos in ["top", "right"]:
        ax_correlations.spines[pos].set_visible(False)
    if i != len(model_names) - 1:
        plt.setp(ax_correlations.get_xticklabels(), visible=False)
    ax_correlations.set_xlim(-0.5, n_layers)
    ax_correlations.set_xticks(np.arange(n_layers))
    ax_correlations.yaxis.set_major_locator(MultipleLocator(0.2))
    ax_correlations.yaxis.set_minor_locator(MultipleLocator(0.1))
    ax_correlations.set_ylim(-0.8, 0.9)
    ax_correlations.set_ylabel("correlation")
    ax_correlations.yaxis.set_label_coords(-0.5, 0.5)
    ax_correlations.yaxis.grid(which="major", alpha=0.5)
    if i == len(model_names) - 1:
        ax_correlations.set_xlabel("layer")
        ax_correlations.xaxis.set_label_coords(-0.2, -0.05)

all_pvalues = [np.array(p) for p in all_pvalues]
pvalues = fdrcorrection(np.hstack([p.ravel() for p in all_pvalues]))[1]
j = 0
for i, p in enumerate(all_pvalues):
    all_pvalues[i] = pvalues[j : j + p.size].reshape(p.shape)
    j += p.size

for i in range(len(all_pvalues)):
    for j in range(len(all_pvalues[i])):
        ax = axes_response_patterns[i][j]
        for k in range(len(all_pvalues[i][j])):
            p = all_pvalues[i][j][k]
            ymax = all_ymax[i][j][k]
            effect_size = all_effect_size[i][j][k]
            if p < 0.05:
                y_step = 0.1
                y_min = min(ymax, 2.3)
                y_max = y_min + y_step
                ax.plot(
                    [k / 3 - 0.05, k / 3 - 0.05],
                    [y_min, y_max],
                    color="#666",
                    linewidth=1,
                )
                ax.plot(
                    [(k + 1) / 3 - 0.15, (k + 1) / 3 - 0.15],
                    [y_min, y_max],
                    color="#666",
                    linewidth=1,
                )
                ax.plot(
                    [k / 3 - 0.05, (k + 1) / 3 - 0.15],
                    [y_max, y_max],
                    color="#666",
                    linewidth=1,
                )
                label = f"{effect_size:.2f}"
                if label[0] == "0":
                    label = label[1:]
                label = label[:3]
                ax.text(
                    (k / 3) + 0.1,
                    y_max,
                    label,
                    ha="center",
                    va="bottom",
                    fontsize=10,
                    color="#666",
                )

plt.subplots_adjust(
    top=0.95,
    bottom=0.025,
    left=-0.03,
    right=0.995,
    hspace=0.08,
    wspace=0.05,
)

ax_legend = fig.add_axes((0.1, 0.95, 0.9, 0.05))
ax_legend.set_xlim(0, 1)
ax_legend.set_ylim(0, 1)
ax_legend.set_axis_off()
x = 0.05
for stimulus_type, color in zip(stimulus_types, colors):
    obj = ax_legend.text(
        x,
        0.5,
        stimulus_type,
        ha="left",
        va="center_baseline",
        fontsize=10,
        color=color,
        bbox=dict(boxstyle="square", ec=color, fc="white"),
    )
    bbox = ax_legend.transData.inverted().transform(
        obj.get_window_extent(fig.canvas.get_renderer())
    )
    x = bbox[1, 0] + 0.015

x = 0.83
for i, landmark in enumerate(["Type I", "Type II", "n400m"]):
    obj = ax_legend.text(
        x,
        0.5,
        landmark,
        ha="left",
        va="center_baseline",
        fontsize=10,
        color=f"C{i+6}",
        bbox=dict(boxstyle="square", ec=f"C{i+6}", fc="white"),
    )
    bbox = ax_legend.transData.inverted().transform(
        obj.get_window_extent(fig.canvas.get_renderer())
    )
    x = bbox[1, 0] + 0.015


if not os.path.exists(fig_fname) or overwrite:
    plt.savefig(fig_fname)
