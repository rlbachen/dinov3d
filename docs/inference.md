# Inferance [WIP]

We plan to implement a more robust way to run inference without having to create
a new dataset and preprocess each time, but for now the best way to do
inference is as follows.

First follow the [preprocessing](./usage/preprocessing.md) instructions on your data first.
Skull stripping is not required, but may help performance in some tasks.

To load the model and set up a dataset just use the following functions

```
from dinov3.inferance import load_teacher_model_for_inference, make_inference_dataset, make_inference_dataloader

model, embed_dim = load_teacher_model_for_inferance("/path/to/config.yaml", "/path/to/checkpoint.pth")

dataset = made_inference_dataset("/path/to/dataset.json", "/path/to/dataset/folder")

dataloader = make_inference_dataloader(dataset, num_workers=4)
```

You can then iterate though the dataset to do inference:

```
outs = []
device = torch.device("cuda") # use whatever you need here
subject_shape = 0
for subject in dataloader:
    subject_tensor = subject["image"].to(device)
    subject_shape = subject_tensor.shape
    out = model.forward_features(subject_tensor)
    outs.append(out)
```

Some notes on what is going on here:

The make_inference_dataset adds a layer of [Monai](https://project-monai.github.io/) transforms which resample the
scan to isotropic voxel spacing, as is done during training, and which pad the image
to make sure every dim is divisible by the patch size. Padding is done by filling in with the lowest
value present in the image. For some images it might make sense to pad with zeros or another value instead.


This means that not every scan will be resampled to the same output shape, so if your dataset
contains heterogenous scan shapes, you will need to use a batch size of 1.


