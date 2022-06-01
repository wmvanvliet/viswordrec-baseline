"""
Use Captum to plot the pixel contributions to the output.
"""
import torch
import numpy as np
import pandas as pd
import mkl
mkl.set_num_threads(4)
from captum.attr import DeepLift
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

import network
import dataloader

# Path where you downloaded the OSF data to: https://osf.io/nu2ep/
data_path = './data'

# Whether to overwrite the existing figure
overwrite = False

classes = dataloader.TFRecord(f'{data_path}/training_datasets/epasana-10kwords').classes
classes.append(pd.Series(['noise'], index=[10000]))

# Load the TIFF images presented in the MEG experiment and apply the
# ImageNet preprocessing transformation to them.
stimuli = pd.read_csv(f'{data_path}/stimuli.csv')
preproc = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])
images = []
for fname in tqdm(stimuli['tif_file'], desc='Reading images'):
    with Image.open(f'{data_path}/stimulus_images/{fname}') as orig:
        image = Image.new('RGB', (224, 224), '#696969')
        image.paste(orig, (12, 62))
        image = preproc(image).unsqueeze(0)
        images.append(image)
images = torch.cat(images, 0)

# Load the model and feed through the images
checkpoint = torch.load(f'{data_path}/models/vgg11_first_imagenet_then_epasana-10kwords_noise.pth.tar', map_location='cpu')
model = network.VGG11.from_checkpoint(checkpoint, freeze=True)
model.eval()
model_output = model(images).detach().numpy()

layer_names = [
    'conv1',
    'conv1_relu',
    'conv2',
    'conv2_relu',
    'conv3',
    'conv3_relu',
    'conv4',
    'conv4_relu',
    'conv5',
    'conv5_relu',
    'fc1',
    'fc1_relu',
    'fc2',
    'fc2_relu',
    'word',
    'word_relu',
]

# Translate output of the model to text predictions
predictions = stimuli.copy()
predictions['predicted_text'] = classes[model_output.argmax(axis=1)].values
predictions['predicted_class'] = model_output.argmax(axis=1)

def plot_attributions(img, word=None, cl=None, rotation=0, ax=None, scale=.12):
    if cl is None and word is not None:
        cl = np.flatnonzero(classes == word)[0]
    attributions = DeepLift(model).attribute(
        img.unsqueeze(0), 0, target=int(cl),
        return_convergence_delta=False)
    attributions = attributions.detach().numpy().mean(axis=1)
    if ax is None:
        fig, ax = plt.subplots(1, 1)
    vmax = scale
    vmin = -vmax
    ax.imshow(attributions[0][75:150], vmin=vmin, vmax=vmax, cmap='RdBu_r')
    ax.set_axis_off()
    print(np.abs(attributions).max())
    if word is not None:
        ax.set_title(f'{word}->{classes[cl]}', fontsize=10)
    elif cl is not None:
        ax.set_title(classes[cl], fontsize=10)
    return ax

sel = predictions.query('type=="word"')
fig, axes = plt.subplots(15, 8, figsize=(14, 12))
for ax in axes.flat:
    ax.set_axis_off()
for img, word, cl, ax in tqdm(zip(images[sel.index], sel.text, sel.predicted_class, axes.flat), total=len(sel)):
    plot_attributions(img, word, ax=ax, cl=cl, scale=0.15)
plt.tight_layout()

sel = predictions.query('type=="pseudoword"')
fig, axes = plt.subplots(15, 8, figsize=(14, 12))
for ax in axes.flat:
    ax.set_axis_off()
for img, word, cl, ax in tqdm(zip(images[sel.index], sel.text, sel.predicted_class, axes.flat), total=len(sel)):
    plot_attributions(img, word, ax=ax, cl=cl, scale=0.06)
plt.tight_layout()

sel = predictions.query('type=="consonants"')
fig, axes = plt.subplots(15, 8, figsize=(14, 12))
for ax in axes.flat:
    ax.set_axis_off()
for img, word, cl, ax in tqdm(zip(images[sel.index], sel.text, sel.predicted_class, axes.flat), total=len(sel)):
    plot_attributions(img, word, ax=ax, cl=cl, scale=0.06)
plt.tight_layout()

sel = predictions.query('type=="symbols"')
fig, axes = plt.subplots(15, 8, figsize=(14, 12))
for ax in axes.flat:
    ax.set_axis_off()
for img, cl, ax in tqdm(zip(images[sel.index], sel.predicted_class, axes.flat), total=len(sel)):
    plot_attributions(img, ax=ax, cl=cl, scale=0.15)
plt.tight_layout()

sel = predictions.query('type=="noisy word"')
fig, axes = plt.subplots(15, 8, figsize=(14, 12))
for ax in axes.flat:
    ax.set_axis_off()
for img, cl, ax in tqdm(zip(images[sel.index], sel.predicted_class, axes.flat), total=len(sel)):
    plot_attributions(img, ax=ax, cl=cl, scale=0.06)
plt.tight_layout()
