"""
Run the MEG experiment stimuli through the model and record the mean activity
within each layer.

Requires a lot of RAM (more than 32GB)!
"""

import os
import pickle

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import dataloaders
import networks

# Path where you downloaded the OSF data to: https://osf.io/nu2ep/
data_path = "./data"

# Whether to overwrite the existing data
overwrite = False

# The model to perform the analysis on.
model_name = "vgg11stochastic_first_imagenet_then_10kwords-freq"

# Whether to reset back norm running stats. You want to turn this on when
# evaluating the amount of activity within the layers in response to wildly
# different stimulus images. In our study, we have this turned on because the
# noise embedded words are very different from the other stimuli. When
# evaluating stimulus images that are more similar, or evaluating raw
# classification accuracy, you want to turn this off.
reset_batchnorm_stats = True

# Get the images that were presented during the MEG experiment
stimuli = pd.read_csv(f"{data_path}/stimuli.csv")

# Transform the images to something suitable for feeding through the model.
# This is the same transform as used in train_net.py during the training of
# the model.
preproc = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

# Load the PNG images presented in the MEG experiment and apply the
# transformation to them. Also keep track of the filesizes, as this is a decent
# measure of visual complexity and perhaps useful in the RSA analysis.
images = []
for fname in tqdm(stimuli["tif_file"], desc="Reading images"):
    with Image.open(f"{data_path}/stimulus_images/{fname}") as orig:
        image = Image.new("RGB", (224, 224), "#696969")
        image.paste(orig, (11, 63))
        image = preproc(image).unsqueeze(0)
        images.append(image)
images = torch.cat(images, 0)

# Determine parameters of the model from the filename
arch, rest = model_name.split("_", 1)
if "-nl" in arch:
    arch, noise_level = arch.split("-nl")
    noise_level = float(noise_level)
else:
    noise_level = 0.1
if "-cs" in arch:
    arch, classifier_size = arch.split("-cs")
    classifier_size = int(classifier_size)
else:
    classifier_size = 4096
imagenet = (
    rest.startswith("first_imagenet_then_")
    or rest.startswith("first_imagenet-1k_then_")
    or rest.startswith("first_imagenet-1k0_then_")
)
if imagenet:
    rest = rest.split("_then_", 1)[1]
num = "_iter-" in rest
if num:
    rest, num = rest.split("_iter-", 1)
    num = int(num)
epoch = "_epoch-" in rest
if epoch:
    rest, epoch = rest.split("_epoch-", 1)
    epoch = int(epoch)
dataset = rest
if dataset == "imagenet":
    dataset = "imagenet-1k"
if dataset == "random":
    dataset = "10kwords-freq-no-noise"
print(
    f"{arch=} {dataset=} {imagenet=} {noise_level=} {classifier_size=} {num=} {epoch=}"
)
classes = dataloaders.WebDataset(f"{data_path}/training_datasets/{dataset}").classes

# Load information about the dataset used to train the model
metadata = pd.read_csv(
    f"{data_path}/training_datasets/{dataset}/train.csv", index_col=0
)
classes = metadata.groupby("label").agg("first")["text"]

# Load the model and feed through the images
checkpoint = torch.load(f"{data_path}/models/{model_name}.pth.tar", map_location="cpu")
model = (
    getattr(networks, arch)
    .from_checkpoint(checkpoint, classifier_size=classifier_size)
    .eval()
)

print("=> resetting tracking batchnorm running stats")
if reset_batchnorm_stats:
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None

layer_outputs = model.get_layer_activations(images)
layer_activity = []
for output in layer_outputs:
    layer_activity.append(output)
mean_activity = np.array(
    [
        np.sqrt(np.square(a.reshape(len(stimuli), -1)).sum(axis=1))
        for a in layer_activity
    ]
)

# Store the activity
layer_names = model.readout_layer_names
fname = f"{data_path}/model_layer_activity/{model_name}_mean_layer_activity.pkl"
if not os.path.exists(fname) or overwrite:
    with open(fname, "wb") as f:
        pickle.dump(
            dict(layer_names=model.readout_layer_names, mean_activity=mean_activity), f
        )

# Translate output of the model to text predictions
predictions = stimuli.copy()
predictions["predicted_text"] = classes[
    layer_activity[model.readout_layer_names.index("word_relu")].argmax(axis=1)
].values
predictions["predicted_class"] = layer_activity[
    model.readout_layer_names.index("word_relu")
].argmax(axis=1)

predictions_fname = f"{data_path}/predictions/{model_name}-predictions.csv"
if not os.path.exists(predictions_fname) or overwrite:
    predictions.to_csv(predictions_fname)

# How many of the word stimuli did we classify correctly?
word_predictions = predictions.query('type=="word"')
n_correct = (
    word_predictions["text"].str.lower()
    == word_predictions["predicted_text"].str.lower()
).sum()
accuracy = n_correct / len(word_predictions)
print(
    f"Word prediction accuracy: {n_correct}/{len(word_predictions)} = {accuracy * 100:.1f}%"
)
