# Preprocessing

For preprocessing, we simply normalize by z-score and fill any empty volumes,
and make sure the files are uncompressed when saved for faster loading.

Scaling to an isotropic voxel spacing is done during inference.

Monai, the library used for data loading and augmentation uses a simple json format
for storing the dataset, which here contains the voxel spacing and size of the
scans.

While in theory we should be able to pull this information from the Nifti headers
during runtime, in practice these Nifti headers can be missing or incorrect, so
instead we gather this information from the scans themselves, and store it in the json.

So for preprocessing this looks like the following steps:

## Generate a dataset Json

This can be done using the generate_data_json.py script, which has the following arguments:
- ` --bucket s3_bucket_name`
- ` --prefix any prefix you which to ignore in the s3 file paths`
- ` --local_dir path to dataset if `
- ` --full_path Preserve file structure of dataset, default false `
- ` --num_cpus for parallel usage, runs without ray if set to 1, default 16 `
- ` --test for testing, only runs on 1000 scans, default false`

## Preprocess Dataset

Once you have your dataset Json, you can now run preprocessing with the
preproccess_dataset.py script, with the following args:
- `--input_s3 s3 bucket for input`
- `--output_s3 s3 bucket for output`
- `--input_dir local dir for input`
- `--output_dir local dir for output`
- `--dataset_json path to dataset json`
- `--num_workers number of workers for ray`

This currently does not support running with ray, which should probably be added later XXX

Note that you cannot use both input_s3 and input_dir at the same time for obvious reasons,
but it does support output to both s3 and local simultaneously.

## Final dataset Json

In theory the dataset Json should be the same between the original and preprocessed dataset,
with possibly a change of file extension from .nii.gz to .nii, but you can also just
run the script again to get a final dataset Json.

This should probably be streamlined later XXX.