import os
import sys
from collections import Counter
from io import BytesIO

import cv2
import Levenshtein
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LinearRegression
from torchvision import transforms
from tqdm import tqdm

import networks

sys.path.append("training_datasets")
import render_stimulus

# Path where you downloaded the OSF data to: https://osf.io/nu2ep/
data_path = "./data"

# Whether to overwrite the existing figure
overwrite = False

# The filename for the produced figure
fig_fname = "figures/post_hoc_exploration.pdf"

# Model to evaluate
model_name = "vgg11stochastic_first_imagenet_then_10kwords-freq"
layer_names = ["conv1", "conv2", "conv3", "conv4", "conv5", "fc1", "fc2", "output"]


def orthographic_neighbourhood(word, words, max_dist):
    """Compute orthographic neighbourhood size.

    Parameters
    ----------
    word : str
        The word to compute the neighbourhood size for.
    words : list of str
        The vocabulary.
    max_dist : int
        The maximum Levenshtein distance to regard a word an orthographic neighbour.

    Returns
    -------
    neighbourhood_size : int
        The orthographic neighbourhood size of the word.
    """
    if type(word) is not str:
        return np.NaN
    return sum(
        [
            int(Levenshtein.distance(word, word2) <= max_dist)
            for word2 in words
            if word != word2
        ]
    )


def correlate(df, var, covars=None, interest=None):
    """Perform correlation with possible correction for covariates.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataframe containing the data.
    var : str
        The column in the dataframe to use as independant variable.
    covars : list of str | None
        An optional list of column names to be regressed out before doing the
        correlation. The squared values will also be regressed out.
    interest : list of str | None
        The column names of the dependant variables. If ``None``, use all the layers of
        the model.

    Returns
    -------
    corr : float
        The Pearson correlation value.
    """
    if interest is None:
        interest = layer_names
    columns = interest + [var]
    if covars is not None:
        df_p = df[columns + covars].copy()
        for covar in covars:
            df_p[f"{covar}2"] = df_p[covar] ** 2
        covars += [f"{covar}2" for covar in covars]
        df_p = df_p.dropna()
        df_p[columns] = df_p[columns] - LinearRegression().fit(
            df_p[covars], df_p[columns]
        ).predict(df_p[covars])
    else:
        df_p = df[columns].copy().dropna()
    return df_p.corr().loc[interest, var]


## Annotate the stimulus set with various stuff
training_set = pd.read_csv(
    f"{data_path}/training_datasets/10kwords-freq/train.csv", index_col=0
)
df = training_set.groupby("text").agg(dict(stimulus="first", label="count"))
df = df.reset_index()
df.columns = ["word", "text", "count"]

# Orthographic neighbourhood size
df["orth1"] = [
    orthographic_neighbourhood(word, df.word, max_dist=1)
    for word in tqdm(df["word"], desc="Orth. neighb. size")
]
df = df.set_index("word")

# Word frequency
freqs = pd.read_csv(
    f"{data_path}/training_datasets/lemma_frequencies.freq",
    sep=" ",
    index_col=0,
)
df = df.join(freqs, on="word")
df["sqrtfreq"] = df["freq"] ** 0.2
df["logfreq"] = np.log(df["freq"])

# Word length
df["len"] = df.text.str.len()

# Letter frequency
lettercounts = Counter()
for word, data in df.iterrows():
    lettercounts.update(list(data["text"]) * data["count"])
lettercounts = pd.DataFrame(
    dict(count=lettercounts.values()), index=lettercounts.keys()
)
bigramcounts = Counter()
for word, data in df.iterrows():
    bigramcounts.update(
        ["".join(bigram) for bigram in zip(data["text"][:-1], data["text"][1:])]
        * data["count"]
    )
bigramcounts = pd.DataFrame(
    dict(count=bigramcounts.values()), index=bigramcounts.keys()
)
bigramcounts["lettercount"] = [
    sum([lettercounts.at[letter, "count"] for letter in list(bigram)])
    for bigram in bigramcounts.index
]

letter_freqs = []
bigram_freqs = []
for word in df.text:
    f = np.mean([lettercounts["count"].get(letter, 0) for letter in list(word)])
    letter_freqs.append(f)
    f = np.mean(
        [
            bigramcounts["count"].get("".join(bigram), 0)
            for bigram in zip(word[:-1], word[1:])
        ]
    )
    bigram_freqs.append(f)
df["letterfreq"] = letter_freqs
df["bigramfreq"] = bigram_freqs

# Load model
m = networks.vgg11stochastic.from_checkpoint(
    torch.load(
        f"{data_path}/models/{model_name}.pth.tar",
        map_location="cuda",
        weights_only=False,
    )
)
m.cuda().eval()

transform = transforms.Compose(
    [
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

## Training set words
all_acts = [list() for _ in range(8)]
edges = []
for i, word in enumerate(df.text):
    buf = render_stimulus.render(word, f"{data_path}/fonts/arial.ttf", 23, 0, 0)
    img = Image.open(BytesIO(bytes(buf)))
    edges.append(cv2.Canny(np.array(img)[:, :, ::-1], 100, 200).sum())
    with torch.no_grad():
        for j, act in enumerate(
            m.get_layer_activations(transform(img).unsqueeze(0).cuda(), verbose=False)
        ):
            all_acts[j].append(np.sqrt(np.sum(act**2)))
    print(f"{i:05d}/{len(df)}", word)
all_acts = np.array(all_acts)
df[["conv1", "conv2", "conv3", "conv4", "conv5", "fc1", "fc2", "output"]] = all_acts.T
df["edges"] = edges

## Pure noise
rng = np.random.RandomState(0)
chosen_noise_levels = rng.rand(1_000)
edges = list()
all_acts = [list() for _ in range(8)]
for noise_level in tqdm(chosen_noise_levels, desc="Noise levels"):
    buf = render_stimulus.render("", f"{data_path}/fonts/arial.ttf", 23, 0, noise_level)
    img = Image.open(BytesIO(bytes(buf)))
    edges.append(cv2.Canny(np.array(img)[:, :, ::-1], 100, 200).sum())
    with torch.no_grad():
        for j, act in enumerate(
            m.get_layer_activations(transform(img).unsqueeze(0).cuda(), verbose=False)
        ):
            all_acts[j].append(np.sqrt(np.sum(act**2)))
all_acts = np.array(all_acts)
df_noise = pd.DataFrame(dict(noise_level=chosen_noise_levels))
df_noise[["conv1", "conv2", "conv3", "conv4", "conv5", "fc1", "fc2", "output"]] = (
    all_acts.T
)
df_noise["edges"] = edges

## Noise-embedded word
rng = np.random.RandomState(0)
df_noisy_word = df[df.text.str.len() == 8].copy()
df_noisy_word["noise_level"] = rng.rand(len(df_noisy_word)) * 0.5
edges = list()
all_acts = [list() for _ in range(8)]
for word, stimulus in df_noisy_word.iterrows():
    buf = render_stimulus.render(
        stimulus["text"], f"{data_path}/fonts/arial.ttf", 23, 0, stimulus["noise_level"]
    )
    img = Image.open(BytesIO(bytes(buf)))
    edges.append(cv2.Canny(np.array(img)[:, :, ::-1], 100, 200).sum())
    with torch.no_grad():
        for j, act in enumerate(
            m.get_layer_activations(transform(img).unsqueeze(0).cuda(), verbose=False)
        ):
            all_acts[j].append(np.sqrt(np.sum(act**2)))
    print(stimulus["text"], stimulus["noise_level"])
all_acts = np.array(all_acts)
df_noisy_word[["conv1", "conv2", "conv3", "conv4", "conv5", "fc1", "fc2", "output"]] = (
    all_acts.T
)
df_noisy_word["edges"] = edges

## Symbols
rng = np.random.RandomState(0)
symbols = {
    "s": "\u25fb",  # Square
    "o": "\u25cb",  # Circle
    "^": "\u25b3",  # Triangle up
    "v": "\u25bd",  # Triangle down
    "d": "\u25c7",  # Diamond
}
chosen_symbols = [
    "".join(rng.choice(list(symbols.values()), length)) for length in df["len"]
]
edges = list()
all_acts = [list() for _ in range(8)]
for text in chosen_symbols:
    buf = render_stimulus.render(
        text, f"{data_path}/fonts/DejaVuSansMono.ttf", 30, 0, 0
    )
    img = Image.open(BytesIO(bytes(buf)))
    edges.append(cv2.Canny(np.array(img)[:, :, ::-1], 100, 200).sum())
    with torch.no_grad():
        for j, act in enumerate(
            m.get_layer_activations(transform(img).unsqueeze(0).cuda(), verbose=False)
        ):
            all_acts[j].append(np.sqrt(np.sum(act**2)))
    print(text)
all_acts = np.array(all_acts)
df_symbols = df.copy()
df_symbols[["conv1", "conv2", "conv3", "conv4", "conv5", "fc1", "fc2", "output"]] = (
    all_acts.T
)
df_symbols["symbols"] = chosen_symbols
df_symbols["edges"] = edges

df_word_symbols = pd.concat([df, df_symbols], ignore_index=True)
df_word_symbols["is_word"] = np.hstack((np.ones(10_000), np.zeros(10_000)))

## Rendering at different sizes
words_len8 = [w for w in df.text if len(w) == 8][:1000]

all_acts = [list() for _ in range(8)]
chosen_words = []
chosen_sizes = []
edges = []
for i, word in enumerate(words_len8):
    for size in np.linspace(14, 32, 10):
        chosen_words.append(word)
        chosen_sizes.append(size)
        buf = render_stimulus.render(
            word.replace("#", "").upper(), f"{data_path}/fonts/arial.ttf", size, 0, 0
        )
        img = Image.open(BytesIO(bytes(buf)))
        edges.append(cv2.Canny(np.array(img)[:, :, ::-1], 100, 200).sum())
        with torch.no_grad():
            for j, act in enumerate(
                m.get_layer_activations(
                    transform(img).unsqueeze(0).cuda(), verbose=False
                )
            ):
                all_acts[j].append(np.sqrt(np.sum(act**2)))
    print(f"{i:05d}/{len(words_len8)}", word)
all_acts = np.array(all_acts)

df_sizes = pd.DataFrame(dict(text=chosen_words, size=chosen_sizes))
df_sizes[["conv1", "conv2", "conv3", "conv4", "conv5", "fc1", "fc2", "output"]] = (
    all_acts.T
)
df_sizes["edges"] = edges
df_sizes = df_sizes.join(
    df[["text", "freq", "sqrtfreq", "logfreq", "letterfreq", "bigramfreq"]].set_index(
        "text"
    ),
    on="text",
)

## Rendering in different fonts
font1 = f"{data_path}/fonts/comic.ttf"
font2 = f"{data_path}/fonts/impact.ttf"

all_acts = [list() for _ in range(8)]
edges = []
chosen_fonts = []
chosen_text = []
for i, word in enumerate(df.text):
    for chosen_font, font_file in enumerate([font1, font2], 1):
        buf = render_stimulus.render(word, font_file, 23, 0, 0)
        img = Image.open(BytesIO(bytes(buf)))
        edges.append(cv2.Canny(np.array(img)[:, :, ::-1], 100, 200).sum())

        with torch.no_grad():
            for j, act in enumerate(
                m.get_layer_activations(
                    transform(img).unsqueeze(0).cuda(), verbose=False
                )
            ):
                all_acts[j].append(np.sqrt(np.sum(act**2)))
        print(f"{i:05d}/{len(df)}", word, font_file)
        chosen_fonts.append(chosen_font)
        chosen_text.append(word)
all_acts = np.array(all_acts)

df_fonts = pd.DataFrame(dict(text=chosen_text, font=chosen_fonts))
df_fonts[["conv1", "conv2", "conv3", "conv4", "conv5", "fc1", "fc2", "output"]] = (
    all_acts.T
)
df_fonts["edges"] = edges
df_fonts = df_fonts.join(
    df[["text", "freq", "sqrtfreq", "logfreq", "letterfreq", "bigramfreq"]].set_index(
        "text"
    ),
    on="text",
)

## Random letter strings
rng = np.random.RandomState(0)
chosen_words = [
    "".join(rng.choice(list("ABDEFGHIJKLMNOPRSTUVYÄÖ"), 8, replace=True))
    for _ in range(10000)
]
pngsizes = []
edges = []
all_acts = [list() for _ in range(8)]
for i, word in enumerate(chosen_words):
    buf = render_stimulus.render(word, f"{data_path}/fonts/arial.ttf", 23, 0, 0)
    pngsizes.append(len(buf))
    img = Image.open(BytesIO(bytes(buf)))
    edges.append(cv2.Canny(np.array(img)[:, :, ::-1], 100, 200).sum())
    with torch.no_grad():
        for j, act in enumerate(
            m.get_layer_activations(transform(img).unsqueeze(0).cuda(), verbose=False)
        ):
            all_acts[j].append(np.sqrt(np.sum(act**2)))
    print(f"{i:05d}/{len(chosen_words)}", word)
all_acts = np.array(all_acts)
df_randomletters = pd.DataFrame(dict(word=chosen_words))
df_randomletters[
    ["conv1", "conv2", "conv3", "conv4", "conv5", "fc1", "fc2", "output"]
] = all_acts.T
df_randomletters["pngsize"] = pngsizes
df_randomletters["edges"] = edges
letter_freqs = []
bigram_freqs = []
for word in chosen_words:
    f = np.mean([lettercounts["count"].get(letter, 0) for letter in list(word)])
    letter_freqs.append(f)
    f = np.mean(
        [
            lettercounts["count"].get("".join(bigram), 0)
            for bigram in zip(word[:-1], word[1:])
        ]
    )
    bigram_freqs.append(f)
df_randomletters["letterfreq"] = letter_freqs
df_randomletters["bigramfreq"] = bigram_freqs


## Random bigrams
sel = pd.concat(
    [
        bigramcounts.query("count >= 30_000").sort_values(
            "lettercount", ascending=True
        )[:50],
        bigramcounts.query("count < 10_000").sort_values(
            "lettercount", ascending=False
        )[:50],
    ]
)
rng = np.random.RandomState(0)
chosen_words = ["".join(rng.choice(sel.index, 4, replace=True)) for _ in range(10000)]
pngsizes = []
edges = []
all_acts = [list() for _ in range(8)]
for i, word in enumerate(chosen_words):
    buf = render_stimulus.render(word, f"{data_path}/fonts/arial.ttf", 23, 0, 0)
    pngsizes.append(len(buf))
    img = Image.open(BytesIO(bytes(buf)))
    edges.append(cv2.Canny(np.array(img)[:, :, ::-1], 100, 200).sum())
    with torch.no_grad():
        for j, act in enumerate(
            m.get_layer_activations(transform(img).unsqueeze(0).cuda(), verbose=False)
        ):
            all_acts[j].append(np.sqrt(np.sum(act**2)))
    print(f"{i:05d}/{len(chosen_words)}", word)
all_acts = np.array(all_acts)
df_randombigrams = pd.DataFrame(dict(word=chosen_words))
df_randombigrams[
    ["conv1", "conv2", "conv3", "conv4", "conv5", "fc1", "fc2", "output"]
] = all_acts.T
df_randombigrams["pngsize"] = pngsizes
df_randombigrams["edges"] = edges
letter_freqs = []
bigram_freqs = []
for word in chosen_words:
    f = np.mean([lettercounts["count"].get(letter, 0) for letter in list(word)])
    letter_freqs.append(f)
    f = np.mean(
        [
            bigramcounts["count"].get("".join(bigram), 0)
            for bigram in zip(word[:-1], word[1:])
        ]
    )
    bigram_freqs.append(f)
df_randombigrams["letterfreq"] = letter_freqs
df_randombigrams["bigramfreq"] = bigram_freqs


## Make pretty plot
colors = ["C0", "C0", "C0", "C0", "C0", "C1", "C1", "C2"]
fig, axes = plt.subplots(nrows=2, ncols=6, sharex=True, sharey=True, figsize=(9, 6))

# Straight-up correlations
axes_iter = iter(axes[0, :])

ax = next(axes_iter)
ax.bar(np.arange(8), correlate(df_noise, "noise_level"), color=colors)
ax.axhline(0, color="black", linewidth=1.5)
ax.set_xticks(np.arange(8), layer_names, rotation=90)
ax.set_yticks(np.linspace(-0.6, 1, 17))
ax.set_ylim(-0.6, 1.0)
ax.set_title("Pure noise")
ax.grid(axis="y", which="both")
ax.set_axisbelow(True)

ax = next(axes_iter)
ax.bar(np.arange(8), correlate(df_noisy_word, "noise_level"), color=colors)
ax.axhline(0, color="black", linewidth=1.5)
ax.set_xticks(np.arange(8), layer_names, rotation=90)
ax.set_title("Noisy word")
ax.grid(axis="y", which="both")
ax.set_axisbelow(True)

ax = next(axes_iter)
ax.bar(np.arange(8), correlate(df, "len"), color=colors)
ax.axhline(0, color="black", linewidth=1.5)
ax.set_xticks(np.arange(8), layer_names, rotation=90)
ax.set_title("Num. letters")
ax.grid(axis="y", which="both")
ax.set_axisbelow(True)

ax = next(axes_iter)
ax.bar(np.arange(8), correlate(df_symbols, "len"), color=colors)
ax.axhline(0, color="black", linewidth=1.5)
ax.set_xticks(np.arange(8), layer_names, rotation=90)
ax.set_title("Num. symbols")
ax.grid(axis="y", which="both")
ax.set_axisbelow(True)

ax = next(axes_iter)
ax.bar(np.arange(8), correlate(df_word_symbols, "is_word"), color=colors)
ax.axhline(0, color="black", linewidth=1.5)
ax.set_xticks(np.arange(8), layer_names, rotation=90)
ax.set_title("Word vs. symbols")
ax.grid(axis="y", which="both")
ax.set_axisbelow(True)

ax = next(axes_iter)
ax.bar(np.arange(8), correlate(df_fonts, "font"), color=colors)
ax.axhline(0, color="black", linewidth=1.5)
ax.set_xticks(np.arange(8), layer_names, rotation=90)
ax.set_title("Font family")
ax.grid(axis="y", which="both")
ax.set_axisbelow(True)

axes_iter = iter(axes[1, :])
ax = next(axes_iter)
ax.bar(np.arange(8), correlate(df_sizes, "size"), color=colors)
ax.axhline(0, color="black", linewidth=1.5)
ax.set_xticks(np.arange(8), layer_names, rotation=90)
ax.set_title("Font size")
ax.grid(axis="y", which="both")
ax.set_axisbelow(True)

ax = next(axes_iter)
ax.bar(
    np.arange(8),
    correlate(df_randomletters, "letterfreq", covars=["edges"]),
    color=colors,
)
ax.axhline(0, color="black", linewidth=1.5)
ax.set_xticks(np.arange(8), layer_names, rotation=90)
ax.set_title("Letter freq")
ax.grid(axis="y", which="both")
ax.set_axisbelow(True)

ax = next(axes_iter)
ax.bar(
    np.arange(8),
    correlate(df_randombigrams, "bigramfreq", covars=["letterfreq", "edges"]),
    color=colors,
)
ax.axhline(0, color="black", linewidth=1.5)
ax.set_xticks(np.arange(8), layer_names, rotation=90)
ax.set_title("Bigram freq")
ax.grid(axis="y", which="both")
ax.set_axisbelow(True)

df_len8 = df[df["text"].str.len() == 8]
ax = next(axes_iter)
ax.bar(
    np.arange(8),
    correlate(df_len8, "sqrtfreq", covars=["edges"]),
    color=colors,
)
ax.axhline(0, color="black", linewidth=1.5)
ax.set_xticks(np.arange(8), layer_names, rotation=90)
ax.set_title("Word freq")
ax.grid(axis="y", which="both")
ax.set_axisbelow(True)

ax = next(axes_iter)
ax.bar(
    np.arange(8),
    correlate(
        df_len8,
        "orth1",
        covars=["edges", "sqrtfreq"],
    ),
    color=colors,
)
ax.axhline(0, color="black", linewidth=1.5)
ax.set_xticks(np.arange(8), layer_names, rotation=90)
ax.set_title("Orth. neighb.")
ax.grid(axis="y", which="both")
ax.set_axisbelow(True)

# Hide the remaining axes
for ax in axes_iter:
    ax.set_visible(False)

plt.tight_layout()

if not os.path.exists(fig_fname) or overwrite:
    plt.savefig(fig_fname)


## Samples of stimuli
# for noise_level in [0.0, 0.25, 0.5, 1.0]:
#     buf = render_stimulus.render("", "fonts/arial.ttf", 23, 0, noise_level)
#     img = Image.open(BytesIO(bytes(buf)))
#     img = img.crop([40, 102, 224 - 40, 226 - 100])
#     img.save(f"stimulus_samples/pure_noise_{noise_level:.3f}.png")
#
# for noise_level in [0.0, 0.15, 0.33, 0.5]:
#     buf = render_stimulus.render("AALTO", "fonts/arial.ttf", 23, 0, noise_level)
#     img = Image.open(BytesIO(bytes(buf)))
#     img = img.crop([40, 102, 224 - 40, 226 - 100])
#     img.save(f"stimulus_samples/noisy_word_{noise_level:.3f}.png")
#
# for word in ["JA", "RAHA", "KAHVIA", "AAMIAINEN"]:
#     buf = render_stimulus.render(word, "fonts/arial.ttf", 23, 0, 0)
#     img = Image.open(BytesIO(bytes(buf)))
#     img = img.crop([40, 102, 224 - 40, 226 - 100])
#     img.save(f"stimulus_samples/num_letters_{len(word):d}.png")
#
# for word in ["◻◇", "○▽▽○", "○○◻○◇▽", "○◻○▽▽▽△△▽"]:
#     buf = render_stimulus.render(word, "fonts/DejaVuSansMono.ttf", 30, 0, 0)
#     img = Image.open(BytesIO(bytes(buf)))
#     img = img.crop([40, 104, 224 - 40, 228 - 100])
#     img.save(f"stimulus_samples/num_symbols_{len(word):d}.png")
#
# for word in ["RAHA", "KAHVIA"]:
#     buf = render_stimulus.render(word, "fonts/arial.ttf", 23, 0, 0)
#     img = Image.open(BytesIO(bytes(buf)))
#     img = img.crop([40, 102, 224 - 40, 226 - 100])
#     img.save(f"stimulus_samples/word_vs_symbols_w{len(word):d}.png")
#
# for word in ["○▽▽○", "○○◻○◇▽"]:
#     buf = render_stimulus.render(word, "fonts/DejaVuSansMono.ttf", 30, 0, 0)
#     img = Image.open(BytesIO(bytes(buf)))
#     img = img.crop([40, 104, 224 - 40, 228 - 100])
#     img.save(f"stimulus_samples/word_vs_symbols_s{len(word):d}.png")
#
# for font_family in ["comic", "impact"]:
#     for word in ["RAHA", "KAHVIA"]:
#         buf = render_stimulus.render(word, f"fonts/{font_family}.ttf", 23, 0, 0)
#         img = Image.open(BytesIO(bytes(buf)))
#         img = img.crop([40, 102, 224 - 40, 226 - 100])
#         img.save(f"stimulus_samples/font_family_{font_family}_{word}.png")
#
# for font_size in [13, 20, 26, 32]:
#     buf = render_stimulus.render("SAMMAKKO", "fonts/arial.ttf", font_size, 0, 0)
#     img = Image.open(BytesIO(bytes(buf)))
#     img = img.crop([40, 102, 224 - 40, 226 - 100])
#     img.save(f"stimulus_samples/font_size_{font_size:d}.png")
#
# for i, word in enumerate(["FGBDBBFF", "JOMAÖGMJ", "UTBTJTFK", "AAIATKAV"]):
#     buf = render_stimulus.render(word, "fonts/arial.ttf", 23, 0, 0)
#     img = Image.open(BytesIO(bytes(buf)))
#     img = img.crop([40, 102, 224 - 40, 226 - 100])
#     img.save(f"stimulus_samples/letter_freq_{i:d}.png")
#
# for i, word in enumerate(["AZAÄÖAÖI", "OTPIYATR", "BATETYET", "VISTTTAB"]):
#     buf = render_stimulus.render(word, "fonts/arial.ttf", 23, 0, 0)
#     img = Image.open(BytesIO(bytes(buf)))
#     img = img.crop([40, 102, 224 - 40, 226 - 100])
#     img.save(f"stimulus_samples/bigram_freq_{i:d}.png")
#
# for i, word in enumerate(["TERMIINI", "SUINPÄIN", "KUORSATA", "TAPAHTUA"]):
#     buf = render_stimulus.render(word, "fonts/arial.ttf", 23, 0, 0)
#     img = Image.open(BytesIO(bytes(buf)))
#     img = img.crop([40, 102, 224 - 40, 226 - 100])
#     img.save(f"stimulus_samples/word_freq_{i:d}.png")
#
# for i, word in enumerate(["PUUTUOTE", "VISKAALI", "BRUTAALI", "KIRITTÄÄ"]):
#     buf = render_stimulus.render(word, "fonts/arial.ttf", 23, 0, 0)
#     img = Image.open(BytesIO(bytes(buf)))
#     img = img.crop([40, 102, 224 - 40, 226 - 100])
#     img.save(f"stimulus_samples/orth1_{i:d}.png")
#
# if not os.path.exists(fig_fname) or overwrite:
#     plt.savefig(fig_fname)
