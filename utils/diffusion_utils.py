import torch
import torch.nn.functional as F

def padding_latents_at_front(source_latents, front_padding_num):
    latents_list = []
    for i in range(front_padding_num):
        latents_list.append(source_latents[:, :, [0]])
    latents_list.append(source_latents)
    padded_latents = torch.cat(latents_list, dim=2)
    return padded_latents


def padding_latents_at_end(source_latents, end_padding_num):
    latents_list = [source_latents]
    for i in range(end_padding_num):
        latents_list.append(source_latents[:, :, [-1]])
    padded_latents = torch.cat(latents_list, dim=2)
    return padded_latents


def resize_video_latent(input_latent, target_height, target_width, mode='bilinear', align_corners=False):
    if mode == 'nearest':
        align_corners = None

    batch, channel, frame, h, w = input_latent.shape

    input_latent = input_latent.permute(0, 2, 1, 3, 4)
    video_reshaped = input_latent.view(batch * frame, channel, h, w)
    upsampled = F.interpolate(video_reshaped, size=(target_height, target_width), mode=mode, align_corners=align_corners)
    upsampled = upsampled.view(batch, frame, channel, target_height, target_width)
    upsampled = upsampled.permute(0, 2, 1, 3, 4)

    return upsampled



