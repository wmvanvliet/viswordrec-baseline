# A baseline deep learning model of visual word recognition

This repository contains the model and code related to "van Vliet et al. 2022".


## Generating the training datasets

Pre-build training datasets can be downloaded from OSF.
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
