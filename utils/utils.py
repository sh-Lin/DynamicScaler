import importlib
import os
from datetime import datetime

import numpy as np
import cv2
import torch
import torch.distributed as dist


from dataclasses import dataclass, field
from typing import Optional

@dataclass
class RunArgs:
    config: str = "configs/inference_t2v_freetraj_512_v2.0.yaml"
    seed: int = 321
    video_length: int = 16
    num_partitions: int = 4
    num_inference_steps: int = 16
    prompt_file: str = "prompts/test_prompts.txt"
    new_video_length: int = 50
    num_processes: int = 1
    rank: int = 0
    height: int = 320
    width: int = 512
    save_frames: bool = False
    fps: int = 8
    unconditional_guidance_scale: float = 12.0
    lookahead_denoising: bool = True
    eta: float = 1.0
    output_dir: Optional[str] = None
    use_mp4: bool = False
    output_fps: int = 10


def count_params(model, verbose=False):
    total_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"{model.__class__.__name__} has {total_params*1.e-6:.2f} M params.")
    return total_params


def check_istarget(name, para_list):
    """ 
    name: full name of source para
    para_list: partial name of target para 
    """
    istarget=False
    for para in para_list:
        if para in name:
            return True
    return istarget


def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def load_npz_from_dir(data_dir):
    data = [np.load(os.path.join(data_dir, data_name))['arr_0'] for data_name in os.listdir(data_dir)]
    data = np.concatenate(data, axis=0)
    return data


def load_npz_from_paths(data_paths):
    data = [np.load(data_path)['arr_0'] for data_path in data_paths]
    data = np.concatenate(data, axis=0)
    return data   


def resize_numpy_image(image, max_resolution=512 * 512, resize_short_edge=None):
    h, w = image.shape[:2]
    if resize_short_edge is not None:
        k = resize_short_edge / min(h, w)
    else:
        k = max_resolution / (h * w)
        k = k**0.5
    h = int(np.round(h * k / 64)) * 64
    w = int(np.round(w * k / 64)) * 64
    image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    return image


def setup_dist(args):
    if dist.is_initialized():
        return
    torch.cuda.set_device(args.local_rank)
    torch.distributed.init_process_group(
        'nccl',
        init_method='env://'
    )


def create_dir(prompt=[], path_root="results", project_id=None, project_folder=None):

    dt_now_str = datetime.now().strftime("%m%d_%H-%M-%S")

    project_id = f"{dt_now_str}-{prompt[:20]}" if project_id is None else f"{dt_now_str}-{project_id}"

    if project_folder is not None:
        output_dir = os.path.join(path_root, f"output/{project_folder}/{project_id}")
        tmp_dir = os.path.join(path_root, f"temp/{project_folder}/{project_id}")
    else:
        output_dir = os.path.join(path_root, f"output/{project_id}")
        tmp_dir = os.path.join(path_root, f"temp/{project_id}")

    print("Output > ", output_dir)
    print("Temp > ", tmp_dir)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    return output_dir, tmp_dir
