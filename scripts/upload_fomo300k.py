import os
import argparse
from monai.data import PersistentDataset
import nibabel as nib
import boto3
import tempfile
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