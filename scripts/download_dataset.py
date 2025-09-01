import json
import os
import argparse
from tqdm import tqdm
import boto3

import multiprocessing
from pathlib import Path


# Use __file__ to get the directory of the current script
script_dir = Path(__file__).resolve().parent

parent_dir = script_dir.parent
use_file_obj = True

def download_file(job_arg):
    bucket, file_name, out_dir = job_arg
    s3_client = boto3.client('s3')
    out_pth = os.path.join(out_dir, file_name)
    # ensure parent directory exists
    parent = os.path.dirname(out_pth)
    if parent:
        os.makedirs(parent, exist_ok=True)
    # skip download if file already exists and is non-empty
    if os.path.exists(out_pth) and os.path.getsize(out_pth) > 0:
        return
    if use_file_obj:
        with open(out_pth, "wb") as f:
            s3_client.download_fileobj(bucket,
                                       file_name,
                                       f)
    else:
        s3_client.download_file(bucket, file_name, out_pth)

def main(args):
    dataset_pth = args.dataset_json
    s3_bucket = args.bucket

    with open(dataset_pth, "r") as f:
        dataset = json.load(f)

    # ensure output directory exists
    os.makedirs(args.out_dir, exist_ok=True)

    if args.num_workers < 0:
        for item in tqdm(dataset):
            file_pth = item["image"]
            try:
                download_file((s3_bucket, file_pth, args.out_dir))
            except Exception as e:
                print(f'subject {file_pth} failed with error: {e}')
    if args.num_workers >= 0:

        jobs = [(s3_bucket, subject['image'], args.out_dir) for subject in dataset]

        num_workers = multiprocessing.cpu_count() if args.num_workers == 0 else args.num_workers
        pool = multiprocessing.Pool(num_workers)
        pool.map(download_file, jobs)
        pool.close()
        pool.join()



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--bucket")
    parser.add_argument("--dataset_json", type=str, required=True)
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)