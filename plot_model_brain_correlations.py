import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
from matplotlib.ticker import MultipleLocator

# Path where you downloaded the OSF data to: https://osf.io/nu2ep/
data_path = './data'

model_names = [
    "vgg11_imagenet",
    "vgg11_first_imagenet_then_250words",
    "vgg11stochastic_first_imagenet_then_250words",
    "vgg11stochastic_first_imagenet_then_1kwords",
    "vgg11stochastic_first_imagenet_then_10kwords",
    "vgg11stochastic_first_imagenet_then_10kwords-freq",
]

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

# Compute correlations between each model and the dipole activity
stimuli = pd.read_csv(f"{data_path}/stimuli.csv")
dipole_acts = pd.read_csv(f'{data_path}/dipoles/grand_average_dipole_activation.csv')

ga_corrs = dict()
for model_name in model_names:
    print(model_name)
    with open(f'{data_path}/model_layer_activity/{model_name}_mean_layer_activity_torch19.pkl', 'rb') as f:
        d = pickle.load(f)
        model_activity = pd.DataFrame(
            np.array(d["mean_activity"]).T,
            columns=d["layer_names"],
        )
        data = dipole_acts.join(model_activity)

    ga_corrs[model_name] = (
        data.corr(numeric_only=True, method="pearson")
        .loc[layer_names, landmarks]
        .max()
    )

## Grand-average plot
fig, axes = plt.subplots(nrows=3, figsize=(5.8, 6), sharex=True, sharey=True)
noise_ceils = [0.80, 0.52, 0.46]  # computed on single-subject data and hence not available in release
for landmark, ax, noise_ceil in zip(landmarks, axes.ravel(), noise_ceils):
    ys = np.arange(len(model_names))[::-1]
    ax.barh(
        ys,
        [ga_corrs[model_name].at[landmark] for model_name in model_names],
        height=0.5,
    )
    ax.axvline(noise_ceil, color="#aaa", linewidth=1)
    ax.axvline(0, color='black', linewidth=1)
    ax.set_yticks(ys, model_names)
    ax.set_ylim(-0.5, len(model_names) - 0.5)
    ax.xaxis.set_major_locator(MultipleLocator(0.3))
    ax.xaxis.set_minor_locator(MultipleLocator(0.1))
    ax.tick_params(which="major", length=5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
ax.set_xlabel("correlation")
plt.tight_layout()
plt.savefig("figures/ga_model_brain_correlations.pdf")
