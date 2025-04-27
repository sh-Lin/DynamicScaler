import os
import re

import torch
from torchvision import transforms
from PIL import Image

from pipeline.d_scheduler import T2VTurboScheduler

import torch.nn.functional as F


def _extract_number(filename):
    match = re.match(r'window_(\d+)\.png$', filename)
    if match:
        return int(match.group(1))
    else:
        match = re.search(r'window_(\d+)\.jpg$', filename)
        if match:
            return int(match.group(1))
        else:
            return float('inf')
def _load_and_preprocess_image(image_path, image_size):
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
        transforms.Lambda(lambda x: x * 2.0 - 1.0)
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image)


def encode_image_to_latent_tensor(pretrained_t2v, image_path, image_size):
    img_tensor = _load_and_preprocess_image(image_path, image_size)
    img_latent = pretrained_t2v.encode_first_stage_2DAE(img_tensor.unsqueeze(1).unsqueeze(0).to(dtype=pretrained_t2v.dtype,
                                                                                                device=pretrained_t2v.device))
    # [ c, h, w ] -> [ b, c, t, h, w ]
    return img_latent


def get_img_list_from_folder(image_folder):
    image_files = [f for f in os.listdir(image_folder) if f.endswith('.png') or f.endswith('.jpg')]
    image_files_name_list = sorted(image_files, key=_extract_number)
    image_path_list = [os.path.join(image_folder, image_name) for image_name in image_files_name_list]
    return image_path_list

def encode_images_list_to_latent_tensor(pretrained_t2v, image_folder, image_size, image_path_list=None):

    if image_path_list is None:
        image_path_list = get_img_list_from_folder(image_folder)

    latent_list = []

    for image_path in image_path_list:
        img_tensor = _load_and_preprocess_image(image_path, image_size)
        img_latent = pretrained_t2v.encode_first_stage_2DAE(img_tensor.unsqueeze(1).unsqueeze(0).to(dtype=pretrained_t2v.dtype, device=pretrained_t2v.device))
        latent_list.append(img_latent)
        # [ c, h, w ] -> [ b, c, t, h, w ]

    video_latent = torch.cat(latent_list, dim=2)

    return video_latent


