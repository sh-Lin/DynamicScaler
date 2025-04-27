import cv2
import numpy as np
import torch
from PIL import Image


def load_image_tensor_from_path(image_path: str, height: int, width: int, norm_to_1=True):
    img = Image.open(image_path).convert("RGB")
    rgb_img = np.array(img, np.float32)
    rgb_img = cv2.resize(rgb_img, (width, height), interpolation=cv2.INTER_LINEAR)
    img_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1)  # .float()

    if norm_to_1:
        img_tensor = (img_tensor / 255. - 0.5) * 2  # Normalize to [-1, 1]

    return img_tensor


def mix_latents_with_mask(latent_1, latent_to_add, mask, mix_ratio):

    if len(mask.shape) == 3:
        mask_expanded = mask.unsqueeze(0).unsqueeze(0)  # 变为 [1, 1, 1, H, W]
        mask_expanded = mask_expanded.repeat(latent_1.size(0), latent_1.size(1), latent_1.size(2), 1, 1)  # 扩展为 [B, N, C, H, W]
    elif len(mask.shape) == 5:
        mask_expanded = mask
    else:
        print("noise shape should be [1, H, W] or [B, N, C, H, W]")
        raise NotImplementedError

    weighted_latent_1 = latent_1 * (1 - mix_ratio)
    weighted_latent_to_add = latent_to_add * mix_ratio
    mixed_area = weighted_latent_1 + weighted_latent_to_add

    non_mask_area = latent_1 * (1 - mask_expanded)
    mask_area = mixed_area * mask_expanded

    mixed_latent = non_mask_area + mask_area

    return mixed_latent