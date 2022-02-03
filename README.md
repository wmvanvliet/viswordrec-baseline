# A baseline deep learning model of visual word recognition

This repository contains the model and code to recreate the figures in:

Marijn van Vliet, Oona Rinkinen, Takao Shimizu, Barry Devereux and Riitta Salmelin (2022), "A large-scale computational model to accurately predict early brain activity in response to written words", preprint.

<img src="figures/results.png" width="800">


## Installing model, data and python dependencies

The model is implemented in PyTorch and the architecture is defined in `network.py`. The `train_net.py` script can be used to train it.

The data (including suitable training sets for the model) is stored on OSF: https://osf.io/nu2ep/ (14.8 GB)
By default, the scripts assume the data is placed in a subfolder called `data/`, but each script has a `data_path` variable declared at the top that can be set to another path.

The Python code to generate the figures depends on a number of packages listed in `requirements.txt`. One way to install them is through pip:

```bash
pip install -r requirements.txt
```


## Generating the training datasets

Pre-build training datasets are included in the data that can be downloaded from OSF (https://osf.io/nu2ep).
However, if you wish to build them from scratch, the scripts in `training_datasets/` can be used.
Since these scripts render a million images, the core image generation routine is implemented in [Rust](https://www.rust-lang.org/) instead of Python for extra speed.
There is a Windows version (`render_stimulus.dll`) and Ubuntu Linux version (`render_stimulus.so`) of the compiled Rust code, but if you need to compile for a different platform, you can do:

```bash
cd training_datasets/render_stimulus
cargo build --release
cp target/release/librender_stimulus.so ../render_stimulus.so
```

With the Rust code compiled, the main image generation scripts can be run:

```bash
python training_datasets/construct_epasana-10k.py data/datasets/epasana-10k train
python training_datasets/construct_epasana-10k.py data/datasets/epasana-10k test
python training_datasets/construct_noise.py data/datasets/noise train
python training_datasets/construct_noise.py data/datasets/noise test
```


## Training the model

A pre-trained version of the model is included in the OSF data (`models/vgg11_first_imagenet_then_epasana-10kwords_noise.pth.tar`).
However, if you want to train it from scratch, the `train_net.py` script can be used to do it.
The ImageNet pre-trained version of the model must be used in this case:

```bash
python train_net.py --resume models/vgg11_imagenet.pth.tar
```

The training script will generate a `checkpoint.pth.tar` file after each epoch and a `best_model.pth.tar` file that contains the overall best performing model.


## Generating the figures

The following scripts can be run to reproduce the figures in the paper, and some of the computations:

```
compute_model_brain_comparison_mne.py - Compute correlations between the model and the MNE source estimates
get_model_layer_activity.py           - Run the stimuli through the model and get the mean activation of each layer
plot_dipole_locations.py              - Plot the locations of the ECD groups as used in the analysis.
plot_dipole_timecourses.py            - Plot the grand-average signal timecourses of each ECD group.
plot_model_brain_comparison.py        - Plot the comparison between ECD groups and layers in the model.
plot_model_brain_comparison_mne.py    - Plot the comparison between the MNE source estimates and the layers in the model.
plot_pixel_contributions.py           - Plot the DeepLIFT pixel contribution analysis.
```
