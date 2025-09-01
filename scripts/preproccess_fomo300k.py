import os
import argparse
from monai.data import PersistentDataset
import nibabel as nib
import boto3
import tempfile
import json
import secrets
import string
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
    def __init__(self, s3_source_bucket, s3_target_bucket, dataset_json):

        with open(dataset_json, "r") as f:
            dataset_json = json.load(f)

        self.dataset = PersistentDataset(dataset_json, cache_dir=None, transform=None)
        self.s3_source_bucket = s3_source_bucket
        self.s3_target_bucket = s3_target_bucket
        self.s3 = boto3.client('s3')

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
                # create random local file name to download to (dont wanna deal with long prefix interpreted as file path)
                filename = f"{subject['image'].removesuffix('.gz').removesuffix('.nii')}.nii"
                rand_numbers = ''.join(secrets.choice(string.digits) for _ in range(20))
                local_file = os.path.join(tmpdir, f"{rand_numbers}.nii.gz" if subject['image'].endswith('.gz') else f"{rand_numbers}.nii")

                self.s3.download_file(self.s3_source_bucket, subject['image'], local_file)

                subject['image'] = local_file
                transformed_arr = self.transform(subject)['image']
                orig_img = nib.load(subject["image"])

                out_nifti_img = nib.Nifti1Image(transformed_arr,
                                                affine=orig_img.affine,
                                                header=orig_img.header)

                local_preproc_filename = os.path.join(tmpdir, f'preprocessed_{rand_numbers}.nii')
                nib.save(out_nifti_img, local_preproc_filename)
                self.s3.upload_file(local_preproc_filename, self.s3_target_bucket, filename)

                return True
        except Exception as e:
            print(f'error {e}')

            return False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_s3', type=str)
    parser.add_argument('--output_s3', type=str)
    parser.add_argument('--dataset_json', type=str)
    parser.add_argument('--num_workers', type=int, default=10)

    args = parser.parse_args()
    return args

def main(args):

    with open(args.dataset_json, "r") as f:
        dataset_read_for_len = json.load(f)

    num_subjects = len(dataset_read_for_len)

    items = [i for i in range(num_subjects)]

    actors = [ScanProcessor.remote(args.input_s3,
                                   args.output_s3,
                                   args.dataset_json)
              for _ in range(args.num_workers)]

    futures = []
    for i, item in enumerate(items):
        actor = actors[i % args.num_workers]
        futures.append(actor.__call__.remote(item))

    results = ray.get(futures)

if __name__ == "__main__":
    args = parse_args()
    main(args)