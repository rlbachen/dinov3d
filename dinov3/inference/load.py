import argparse
import math
from pathlib import Path
import yaml

from omegaconf import OmegaConf
import torch

from dinov3.configs import setup_config, setup_job
from dinov3.train.ssl_meta_arch_3d import SSLMetaArch_3d
from dinov3.models import build_model_from_cfg_3d


def load_teacher_model_for_inference(
    config_file,
    teacher_ckpt_path=None,
    device="cuda",
    verbose=False,
    load_ckpt=True,
):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(device)

    cfg = OmegaConf.create(config)
    model, embed_dim = build_model_from_cfg_3d(cfg, only_teacher=True)

    if teacher_ckpt_path:
        model.to_empty(device=device, recurse=True)
        checkpoint = torch.load(teacher_ckpt_path, map_location=device)
        state_dict = checkpoint["teacher"]
        state_dict = {
            k.replace("module.", "").replace("backbone.", ""): v
            for k, v in state_dict.items()
        }

        missing, unexpected = model.load_state_dict(state_dict, strict=False)

        if len(missing) > 0:
            raise Warning(f"Warning some keys are missing: {missing}")
        if verbose:
            print("Missing keys:", missing)
            print("Unexpected keys:", unexpected)
    else:
        print("warning, no ckpt path, random initialization")
        model = model.to_empty(device=device)
        model.init_weights()
        model.to(device)

    return model, embed_dim