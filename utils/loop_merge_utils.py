import os

import torch
import imageio
import numpy as np
from PIL import Image

def tensor2image(batch_tensors):
    img_tensor = torch.squeeze(batch_tensors)  # c,h,w

    image = img_tensor.detach().cpu()
    image = torch.clamp(image.float(), -1., 1.)

    image = (image + 1.0) / 2.0
    image = (image * 255).to(torch.uint8).permute(1, 2, 0)  # h,w,c
    image = image.numpy()
    image = Image.fromarray(image)

    return image


def save_decoded_video_latents(decoded_video_latents, output_path, output_name, fps, save_mp4=True, save_gif=True):

    video_frames_img_list = []

    for frame_idx in range(decoded_video_latents.shape[2]):
        frame_tensor = decoded_video_latents[:, :, [frame_idx]]
        image = tensor2image(frame_tensor)
        video_frames_img_list.append(image)

    print(f"converted {len(video_frames_img_list)} frame tensors")

    if save_mp4:
        mp4_save_path = os.path.join(output_path, f"{output_name}.mp4")
        imageio.mimsave(mp4_save_path, video_frames_img_list, fps=fps)
        print(f"pano video saved to -> {mp4_save_path}")
