import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Path where you downloaded the OSF data to: https://osf.io/nu2ep/
data_path = "./data"

# Whether to overwrite the existing figure
overwrite = False

# The filename for the produced figure
fig_fname = "figures/hyperparameter_tuning.pdf"

landmarks = ["LeftOcci1", "LeftOcciTemp2", "LeftTemp3"]

models = dict(
    bottleneck=[
        (
            "vgg11stochastic-cs1024_first_imagenet_then_10kwords-freq",
            ["conv1_relu", "fc2_relu", "word_relu"],
        ),
        (
            "vgg11stochastic-cs2048_first_imagenet_then_10kwords-freq",
            ["conv1_relu", "fc1_relu", "fc2_relu"],
        ),
        (
            "vgg11stochastic_first_imagenet_then_10kwords-freq",
            ["conv1_relu", "fc1_relu", "word_relu"],
        ),
        (
            "vgg11stochastic-cs8192_first_imagenet_then_10kwords-freq",
            ["conv1_relu", "fc1_relu", "fc2_relu"],
        ),
    ],
    noise_level=[
        (
            "vgg11stochastic-nl0.05_first_imagenet_then_10kwords-freq",
            ["conv1_relu", "fc1_relu", "fc2_relu"],
        ),
        (
            "vgg11stochastic_first_imagenet_then_10kwords-freq",
            ["conv1_relu", "fc1_relu", "word_relu"],
        ),
        (
            "vgg11stochastic-nl0.2_first_imagenet_then_10kwords-freq",
            ["conv1_relu", "fc1_relu", "word_relu"],
        ),
        (
            "vgg11stochastic-nl0.3_first_imagenet_then_10kwords-freq",
            ["conv1_relu", "fc1_relu", "word_relu"],
        ),
        (
            "vgg11stochastic-nl0.4_first_imagenet_then_10kwords-freq",
            ["conv1_relu", "fc1_relu", "word_relu"],
        ),
    ],
    vocab_size=[
        (
            "vgg11stochastic_first_imagenet_then_250words",
            ["conv1_relu", "fc1_relu", "fc2_relu"],
        ),
        (
            "vgg11stochastic_first_imagenet_then_1kwords",
            ["conv1_relu", "fc1_relu", "fc2_relu"],
        ),
        (
            "vgg11stochastic_first_imagenet_then_5kwords",
            ["conv1_relu", "fc1_relu", "fc2_relu"],
        ),
        (
            "vgg11stochastic_first_imagenet_then_10kwords-freq",
            ["conv1_relu", "fc1_relu", "fc2_relu"],
        ),
    ],
    freq_balancing=[
        (
            "vgg11stochastic_first_imagenet_then_10kwords-freq-0.01",
            ["conv1_relu", "fc1_relu", "word_relu"],
        ),
        (
            "vgg11stochastic_first_imagenet_then_10kwords-freq-0.1",
            ["conv1_relu", "fc1_relu", "word_relu"],
        ),
        (
            "vgg11stochastic_first_imagenet_then_10kwords-freq",
            ["conv1_relu", "fc1_relu", "word_relu"],
        ),
        (
            "vgg11stochastic_first_imagenet_then_10kwords-freq-0.3",
            ["conv1_relu", "fc1_relu", "fc2_relu"],
        ),
        (
            "vgg11stochastic_first_imagenet_then_10kwords-freq-0.4",
            ["conv1_relu", "fc1_relu", "word_relu"],
        ),
        (
            "vgg11stochastic_first_imagenet_then_10kwords-freq-0.5",
            ["conv1_relu", "fc1_relu", "word_relu"],
        ),
    ],
)

stimulus_types = ["noisy word", "symbols", "consonants", "word", "pseudoword"]
colors = ["#1f77b4", "#9467bd", "#ff7f0e", "#d62728", "#2ca02c"]

noise_ceiling = pd.read_csv(f"{data_path}/dipoles/noise_ceiling.csv", index_col=0)

# Compute correlations between each model and the dipole activity
stimuli = pd.read_csv(f"{data_path}/stimuli.csv")
ga_dip_activation = pd.read_csv(
    f"{data_path}/dipoles/grand_average_dipole_activation.csv"
)

colors = ["C6", "C7", "C8"]

fig, axes = plt.subplots(nrows=len(models), sharex=True, figsize=(3, len(models) * 1.5))
for (title, model_list), ax in zip(models.items(), axes.ravel()):
    ga_corrs = dict()
    for model_name, chosen_layers in model_list:
        print(model_name)
        with open(
            f"{data_path}/model_layer_activity/{model_name}_mean_layer_activity_torch19.pkl",
            "rb",
        ) as f:
            d = pickle.load(f)
            model_activity = pd.DataFrame(
                np.array(d["mean_activity"]).T,
                columns=d["layer_names"],
                index=stimuli.tif_file,
            )
            ga_model_activity = ga_dip_activation.join(model_activity, on="tif_file")

        ga_corrs[model_name] = np.diag(
            ga_model_activity.corr(numeric_only=True, method="pearson").loc[
                chosen_layers, landmarks
            ]
        )

    for i in range(3):
        ax.plot(
            [v[i] for v in ga_corrs.values()],
            range(len(model_list) - 1, -1, -1),
            color=colors[i],
            marker="s",
            markersize=7,
        )
    for i, ceil in enumerate(noise_ceiling.ga_ceiling.values):
        ax.axvline(
            ceil,
            color=colors[i],
            zorder=0,
            alpha=0.5,
            linewidth=2,
        )
    ax.set_ylim(-0.5, len(model_list) - 0.5)
    ax.set_yticks(range(len(model_list)))
    ax.set_xticks(np.arange(0, 1, 0.1))
    ax.grid(axis="x")
    ax.spines[["right", "top"]].set_visible(False)
ax.set_xlabel("correlation")
plt.tight_layout()

if not os.path.exists(fig_fname) or overwrite:
    plt.savefig(fig_fname)
