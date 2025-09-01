import os
import argparse
from monai.data import PersistentDataset
import nibabel as nib
import boto3
import tempfile
import secrets
import string
import json
import ray


from monai.transforms import (
    Compose,
    LoadImaged,
    NormalizeIntensityd,
    SignalFillEmptyd,
    ToNumpyd,
    ToNumpy
)

def make_dataset(base_dir, dataset_json, transform):
    with open(dataset_json, "r") as f:
        dataset = json.load(f)

    for subject in dataset:
        subject["image"] = os.path.join(base_dir, subject["image"])

    dataset = PersistentDataset(dataset,cache_dir=None, transform=transform)
    return dataset


@ray.remote
class ScanProcessor:
    def __init__(self, dataset_json, s3_source_bucket = None, s3_target_bucket = None,
                 local_input = None, local_output = None):

        with open(dataset_json, "r") as f:
            dataset_json = json.load(f)

        self.dataset = PersistentDataset(dataset_json, cache_dir=None, transform=None)

        if local_input and s3_source_bucket:
            raise ValueError("cannot use both local and s3 input at the same time")
        if not local_output and not s3_target_bucket:
            raise ValueError("need at least one output target, s3 or local")

        self.local_input = local_input
        self.local_output = local_output
        self.s3_source_bucket = s3_source_bucket
        self.s3_target_bucket = s3_target_bucket
        if self.s3_source_bucket or self.s3_target_bucket:
            self.s3 = boto3.client("s3")
        else:
            self.s3 = None

        self.transform = Compose([
            LoadImaged(keys=["image"]),
            SignalFillEmptyd(keys=["image"]),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            ToNumpyd(keys=["image"])
        ])


    def __call__(self, subject_index):
        subject = self.dataset[subject_index]

        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                filename = f"{subject['image'].removesuffix('.gz').removesuffix('.nii')}.nii"

                if self.local_input is not None:
                    local_file = os.path.join(self.local_input, subject['image'])
                else:
                    rand_numbers = ''.join(secrets.choice(string.digits) for _ in range(20))
                    local_file = os.path.join(tmpdir, f"{rand_numbers}.nii.gz" if subject['image'].endswith('.gz') else f"{rand_numbers}.nii")
                    self.s3.download_file(self.s3_source_bucket, subject['image'], local_file)

                subject['image'] = local_file
                transformed_arr = self.transform(subject)['image']
                orig_img = nib.load(subject["image"])

                out_nifti_img = nib.Nifti1Image(transformed_arr,
                                                affine=orig_img.affine,
                                                header=orig_img.header)

                if self.local_output is not None:
                    local_out = os.path.join(self.local_output, os.path.dirname(filename))
                    os.makedirs(local_out, exist_ok=True)
                    local_preproc_filename = os.path.join(local_out,
                                                          f"preproc_{os.path.basename(filename)}")
                    nib.save(out_nifti_img, local_preproc_filename)
                if self.s3_target_bucket is not None:
                    if self.local_output is not None:
                        upload_path = local_preproc_filename
                    else:
                        upload_path = os.path.join(tmpdir,
                                                f"preproc_{filename}")
                        nib.save(out_nifti_img, upload_path)

                    self.s3.upload_file(upload_path, self.s3_target_bucket,
                                        filename)

                return True
        except Exception as e:
            print(f'error {e}')

            return False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_s3', type=str, default=None)
    parser.add_argument('--output_s3', type=str, default=None)
    parser.add_argument('--input_dir', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--dataset_json', type=str)
    parser.add_argument('--num_workers', type=int, default=10)

    args = parser.parse_args()
    return args

def main(args):


    with open(args.dataset_json, "r") as f:
        dataset_read_for_len = json.load(f)

    num_subjects = len(dataset_read_for_len)

    items = [i for i in range(num_subjects)]

    actors = [ScanProcessor.remote(args.dataset_json,
                                   args.input_s3,
                                   args.output_s3,
                                   args.input_dir,
                                   args.output_dir)
              for _ in range(args.num_workers)]

    futures = []
    for i, item in enumerate(items):
        actor = actors[i % args.num_workers]
        futures.append(actor.__call__.remote(item))

    results = ray.get(futures)

if __name__ == "__main__":
    args = parse_args()
    main(args)