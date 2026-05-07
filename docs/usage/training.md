# Training

The Training process of DINOv3 is nearly identical to that of the original DINOv3,
with a few changes to the config.yaml.

- the addition of a dataset_json argument in train, see preprocessing.md for details
- student.arch should be set to vit3d_xxx, instead of vit_xxx
- rgb_mean is removed as all images are single channel


Once your dataset is preprocessed, you simply run:
``python /dinov3d/dinov3/train/train_3d.py --config-file "path/to/config.yaml" --output-dir /path/to/output``

