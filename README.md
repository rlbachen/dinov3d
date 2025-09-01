# Dinov3D

![PCA visualization of dinov3d on t1 MRI image](./docs/figures/pca_dino_vox.gif)

This codebase is a fork of the [Dinov3 Repo](https://github.com/facebookresearch/dinov3) by the Meta Research group, extended to 3d for use with volumetric medical
imaging data.

We Provide a pretrained 3d VIT model, trained from scratch on over 90K Anatomical MRI
scans from the FOMO300k dataset.

Abstract: XXX

# Weight Availability

The models weight for the trained model, as well as a number of checkpoints are
available on the Hugging Face repo here XXX

# Usage

The repo can be installed locally by simply pip installing the directory after cloning the repo.

Simply run ``git clone git@github.com:asagilmore/dinov3d.git``, and then ``pip install ./path/to/dinov3d``


## Inference
Local usage requires a substantial amount of memory. For high resolution inference,
it is recommended to use a CPU with a large amount of memory, as many GPU's do not have the
required vRAM to support this.




## Dataset Preprocessing

For both inference, or pretraining, it is recommended that you follow the
dataset preprocessing guide [here](docs/usage/preprocessing.md)

