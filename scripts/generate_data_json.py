import os
import nibabel as nib
import json
import argparse
import boto3
import tempfile
from tqdm import tqdm
import ray
import glob
import shutil


@ray.remote
def extract_metadata_remote(key_path_pair, bucket_name=None, full_path=False, use_local=False):
    """Remote function for extracting metadata from a single file"""
    key = key_path_pair.get('key')
    local_path = key_path_pair['local_path']

    # If using local files, do not download from S3
    if not use_local:
        s3 = boto3.client('s3')
        s3.download_file(bucket_name, key, local_path)

    img = nib.load(local_path)
    header = img.header
    spacing = [float(x) for x in header.get_zooms()]
    shape = [int(x) for x in header.get_data_shape()]

    # image field
    if 'image' in key_path_pair:
        image = key_path_pair['image']
    else:
        base_name = os.path.basename(key)
        image = key if full_path else base_name

    # Only delete temp files for S3 mode
    if not use_local:
        os.remove(local_path)

    return {
        'image': image,
        'spacing': spacing,
        'shape': shape,
    }


def extract_metadata(key_path_pair, bucket_name=None, full_path=False, use_local=False):
    key = key_path_pair.get('key')
    local_path = key_path_pair['local_path']

    if not use_local:
        s3 = boto3.client('s3')
        s3.download_file(bucket_name, key, local_path)

    img = nib.load(local_path)
    header = img.header
    spacing = [float(x) for x in header.get_zooms()]
    shape = [int(x) for x in header.get_data_shape()]

    if 'image' in key_path_pair:
        image = key_path_pair['image']
    else:
        base_name = os.path.basename(key)
        image = key if full_path else base_name

    if not use_local:
        os.remove(local_path)

    return {
        'image': image,
        'spacing': spacing,
        'shape': shape,
    }


def build_s3_key_path_pairs(args, temp_dir):
    s3_client = boto3.client('s3')
    s3_resource = boto3.resource('s3')

    bucket = s3_resource.Bucket(args.bucket)
    paginator = s3_client.get_paginator('list_objects_v2')

    key_path_pairs = []

    for page in tqdm(paginator.paginate(Bucket=args.bucket, Prefix=args.prefix), desc="Listing files"):
        for obj in page.get('Contents', []):
            key = obj['Key']
            if not (key.endswith('.nii') or key.endswith('.nii.gz')):
                continue
            local_path = os.path.join(temp_dir, key)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            key_path_pairs.append({'key': key, 'local_path': local_path})

    return key_path_pairs


def build_local_key_path_pairs(local_dir, base_dir):
    # Normalize directories
    local_dir = os.path.abspath(local_dir)
    if base_dir is None:
        base_dir = local_dir
    base_dir = os.path.abspath(base_dir)

    pattern_nii = os.path.join(local_dir, '**', '*.nii')
    pattern_niigz = os.path.join(local_dir, '**', '*.nii.gz')

    files = glob.glob(pattern_nii, recursive=True) + glob.glob(pattern_niigz, recursive=True)

    key_path_pairs = []
    for fpath in files:
        fpath = os.path.abspath(fpath)
        # image path is path minus base_dir
        rel_path = os.path.relpath(fpath, base_dir)
        key_path_pairs.append({
            'local_path': fpath,   # actual on-disk path
            'image': rel_path      # stored in json
        })
    return key_path_pairs


def main(args):
    ray.init(num_cpus=args.num_cpus)

    use_local = args.local_dir is not None

    if use_local:
        key_path_pairs = build_local_key_path_pairs(args.local_dir, args.base_dir)
        if args.output is None:
            # default output name from base_dir or local_dir
            base = os.path.basename(os.path.abspath(args.base_dir or args.local_dir))
            args.output = base + "_dataset.json"
        print(f"Found {len(key_path_pairs)} local NIfTI files...")
    else:
        temp_dir = tempfile.mkdtemp()
        key_path_pairs = build_s3_key_path_pairs(args, temp_dir)

        if args.output is None:
            args.output = args.bucket + "_dataset.json"

        print(f"Processing {len(key_path_pairs)} files from S3 in parallel...")

    if args.test:
        key_path_pairs = key_path_pairs[:1000]
        print(f"Test mode: Processing only {len(key_path_pairs)} files...")

    futures = [
        extract_metadata_remote.remote(
            pair,
            args.bucket,
            full_path=args.full_path,
            use_local=use_local
        )
        for pair in key_path_pairs
    ]

    results = []
    with tqdm(total=len(futures), desc="Extracting metadata") as pbar:
        while futures:
            done_id, futures = ray.wait(futures, num_returns=1)
            result = ray.get(done_id[0])
            results.append(result)
            pbar.update(1)

    # Clean up temp_dir only for S3 mode
    if not use_local:
        shutil.rmtree(temp_dir)

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {args.output}")

    ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generate monai format .json from s3 bucket or local directory"
    )

    # S3 options
    parser.add_argument('--bucket')
    parser.add_argument('--prefix', default='')

    # Local options
    parser.add_argument('--local_dir', help='Base directory to search for NIfTI files')
    parser.add_argument('--base_dir', help='Base directory to strip from paths for image field')

    parser.add_argument('--full_path', type=bool)
    parser.add_argument('--output', default=None)
    parser.add_argument('--num_cpus', type=int, default=16)
    parser.add_argument('--test', action='store_true', help='Run in test mode with only 1000 samples')

    args = parser.parse_args()

    # Basic mode check
    if not args.bucket and not args.local_dir:
        parser.error("You must provide either --bucket (S3 mode) or --local_dir (local mode).")

    main(args)
