
import os
import time
import warnings

import cv2
import numpy as np
import torch
from PIL import Image

from tqdm import trange
from typing import List, Optional, Union, Dict, Any
from diffusers import logging

from diffusers import DiffusionPipeline
from lvdm.models.ddpm3d import LatentVisualDiffusion
from pipeline.scheduler import lvdm_DDIM_Scheduler
from utils.precast_latent_utils import encode_images_list_to_latent_tensor
from utils.shift_window_utils import RingLatent, RingImageTensor
from utils.tensor_utils import mix_latents_with_mask

# from diffusers.schedulers.scheduling_ddim import DDIMScheduler

logger = logging.get_logger(__name__)


class VC2_Pipeline_I2V(DiffusionPipeline):
    def __init__(
            self,
            pretrained_t2v: LatentVisualDiffusion,
            scheduler: lvdm_DDIM_Scheduler,
            model_config: Dict[str, Any] = None,
    ):
        super().__init__()

        self.scheduler: lvdm_DDIM_Scheduler
        self.pretrained_t2v: LatentVisualDiffusion

        self.register_modules(
            pretrained_t2v=pretrained_t2v,
            scheduler=scheduler,
        )
        self.vae = pretrained_t2v.first_stage_model
        self.unet = pretrained_t2v.model.diffusion_model
        self.text_encoder = pretrained_t2v.cond_stage_model

        self.model_config = model_config
        self.vae_scale_factor = 8


    def _load_imgs_from_paths(self, img_path_list: list, height=320, width=512):
        batch_tensor = []
        for filepath in img_path_list:
            _, filename = os.path.split(filepath)
            _, ext = os.path.splitext(filename)
            if ext == '.png' or ext == '.jpg':
                img = Image.open(filepath).convert("RGB")
                rgb_img = np.array(img, np.float32)
                rgb_img = cv2.resize(rgb_img, (width, height), interpolation=cv2.INTER_LINEAR)
                img_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1) # .float()
            else:
                print(f'ERROR: <{ext}> image loading only support format: [mp4], [png], [jpg]')
                raise NotImplementedError
            img_tensor = (img_tensor / 255. - 0.5) * 2
            batch_tensor.append(img_tensor)
        return torch.stack(batch_tensor, dim=0)

    @torch.no_grad()
    def basic_sample_shift_multi_windows(
            self,
            prompt: Union[str, List[str]] = None,
            img_cond_path: Union[str, List[str]] = None,
            height: Optional[int] = 320,
            width: Optional[int] = 512,
            frames: int = 16,
            fps: int = 16,
            guidance_scale: float = 7.5,
            num_videos_per_prompt: Optional[int] = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None, 

            init_panorama_latent: torch.Tensor = None,
            num_windows_w: int = None,
            num_windows_h: int = None,
            num_windows_f: int = None,
            loop_step: int = None,
            pano_image_path: str = None,
            dock_at_h=None,
            latents: Optional[torch.FloatTensor] = None,
            num_inference_steps: int = 4,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",

            merge_renoised_overlap_latent_ratio: float = 1,

            use_skip_time=False,
            skip_time_step_idx=None,
            progressive_skip=False,
            **kwargs
    ):
        unet_config = self.model_config["params"]["unet_config"]
        # 0. Default height and width to unet
        frames = self.pretrained_t2v.temporal_length if frames < 0 else frames

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
            if isinstance(img_cond_path, str):
                img_cond_path = [img_cond_path]
            else:
                assert len(img_cond_path) == 1, "[basic_sample] cond img should have same amount as text prompts"
        elif prompt is not None and isinstance(prompt, list):
            assert len(prompt) == len(img_cond_path), "[basic_sample] cond img should have same amount as text prompts"
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        cond_images = self._load_imgs_from_paths(img_path_list=img_cond_path, height=height, width=width)
        cond_images = cond_images.to(self.pretrained_t2v.device)
        img_emb = self.pretrained_t2v.get_image_embeds(cond_images)
        text_emb = self.pretrained_t2v.get_learned_conditioning(prompt)

        imtext_cond = torch.cat([text_emb, img_emb], dim=1)
        cond = {"c_crossattn": [imtext_cond], "fps": fps}


        # 3.5 Prepare CFG if used
        if guidance_scale != 1.0:
            uncond_type = self.pretrained_t2v.uncond_type
            if uncond_type == "empty_seq":
                prompts = batch_size * [""]
                # prompts = N * T * [""]  ## if is_imgbatch=True
                uc_emb = self.pretrained_t2v.get_learned_conditioning(prompts)
            elif uncond_type == "zero_embed":
                c_emb = cond["c_crossattn"][0] if isinstance(cond, dict) else cond
                uc_emb = torch.zeros_like(c_emb)
            else:
                raise NotImplementedError()

            ## process image embedding token
            if hasattr(self.pretrained_t2v, 'embedder'):
                uc_img = torch.zeros(batch_size, 3, height // self.vae_scale_factor, width // self.vae_scale_factor).to(
                    self.pretrained_t2v.device)
                ## img: b c h w >> b l c
                uc_img = self.pretrained_t2v.get_image_embeds(uc_img)
                uc_emb = torch.cat([uc_emb, uc_img], dim=1)

            if isinstance(cond, dict):
                uncond = {key: cond[key] for key in cond.keys()}
                uncond.update({'c_crossattn': [uc_emb]})
            else:
                uncond = uc_emb
        else:
            uncond = None

        # 4. Prepare timesteps
        self.scheduler.make_schedule(num_inference_steps)  # set_timesteps(num_inference_steps)   # , lcm_origin_steps)

        timesteps = np.flip(self.scheduler.ddim_timesteps)  # [ 999, ... , 0 ]


        if use_skip_time and not progressive_skip:
            timesteps = timesteps[skip_time_step_idx:]
            print(f"skip : {skip_time_step_idx}")


        print(f"[basic_sample_shift_multi_windows] denoise timesteps: {timesteps}")
        print(f"[basic_sample_shift_multi_windows] SKIP {skip_time_step_idx} timesteps {'(progressive)' if progressive_skip else ''}")

        total_steps = self.scheduler.ddim_timesteps.shape[0]

        # 5. Prepare latent variable [pano]
        num_channels_latents = unet_config["params"]["in_channels"]
        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor
        total_shape = (
            batch_size,
            num_channels_latents,
            frames * num_windows_f,
            latent_height * num_windows_h,
            latent_width * num_windows_w,
        )

        if init_panorama_latent is None:

            init_panorama_latent = torch.randn(total_shape, device=device).repeat(batch_size, 1, 1, 1, 1)

            if use_skip_time:

                frame_0_latent = encode_images_list_to_latent_tensor(pretrained_t2v=self.pretrained_t2v,
                                                                     image_folder=None,
                                                                     image_size=(height * num_windows_h, width * num_windows_w),
                                                                     image_path_list=[pano_image_path])
                if progressive_skip:
                    for frame_idx, progs_skip_idx in enumerate(list(reversed(range(skip_time_step_idx)))):
                        # noised_frame_latent = self._add_noise(clear_video_latent=frame_0_latent,
                        #                                       time_step_index=total_steps-progs_skip_idx-1)
                        noised_frame_latent = self.scheduler.re_noise(x_a=frame_0_latent,
                                                                      step_a=0,
                                                                      step_b=total_steps - progs_skip_idx - 1)
                        init_panorama_latent[:, :, [frame_idx]] = noised_frame_latent.clone()

                else:
                    clear_repeat_latent = torch.cat([frame_0_latent] * frames * num_windows_f, dim=2)
                    # init_panorama_latent = self._add_noise(clear_video_latent=clear_repeat_latent,
                    #                                        time_step_index=total_steps-1)
                    init_panorama_latent = self.scheduler.re_noise(x_a=clear_repeat_latent,
                                                                   step_a=0,
                                                                   step_b=total_steps-1)

        else:
            print("[basic_sample_shift_multi_windows] using given init latent")
            assert init_panorama_latent.shape == total_shape, f"[basic_sample_shift_multi_windows] " \
                                                              f"init_panorama_latent shape {init_panorama_latent.shape}" \
                                                              f"does not match" \
                                                              f"desired shape {total_shape}"
            init_panorama_latent = init_panorama_latent.clone()


        panorama_ring_latent_handler = RingLatent(init_latent=init_panorama_latent)
        panorama_ring_latent_denoised_handler = RingLatent(init_latent=torch.zeros_like(init_panorama_latent))

        # define window shift
        image_step_size_w = width // loop_step
        latent_step_size_w = image_step_size_w // self.vae_scale_factor
        if num_windows_w == 1:
            latent_step_size_w = 0

        image_step_size_h = height // loop_step
        latent_step_size_h = image_step_size_h // self.vae_scale_factor
        if num_windows_h == 1:
            latent_step_size_h = 0

        latent_step_size_f = frames // loop_step
        if num_windows_f == 1:
            latent_step_size_f = 0

        assert latent_step_size_f > 0 or num_windows_f == 1, f"[basic_sample_shift_multi_windows] loop_step {loop_step} " \
                                                             f"> frames {frames} while num_windows_f {num_windows_f} > 0"

        total_width = width * num_windows_w
        total_height = height * num_windows_h
        panorama_ring_image_tensor_handler = RingImageTensor(image_path=pano_image_path, height=total_height, width=total_width)

        bs = batch_size * num_videos_per_prompt  # ?

        # 6. DDIM Sampling Loop
        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):

                image_pos_left_start = (i % loop_step) * image_step_size_w
                image_pos_top_start = (i % loop_step) * image_step_size_h

                latent_pos_left_start = (i % loop_step) * latent_step_size_w
                latent_pos_top_start = (i % loop_step) * latent_step_size_h

                latent_frames_begin = (i % loop_step) * latent_step_size_f

                print(f"\n"
                      f"i = {i}, t = {t}")

                # reset denoised mask record
                panorama_ring_mask_handler = RingLatent(init_latent=torch.zeros_like(init_panorama_latent))


                for shift_f_idx in range(num_windows_f):

                    for shift_w_idx in range(num_windows_w):

                        shift_h_idies_list = list(range(num_windows_h))
                        if dock_at_h:
                            shift_h_idies_list = [-100] + [-101] + list(range(num_windows_h))

                        for shift_h_idx in shift_h_idies_list:

                            window_image_left = image_pos_left_start + shift_w_idx * width
                            window_image_right = window_image_left + width
                            window_image_top = image_pos_top_start + shift_h_idx * height
                            window_image_down = window_image_top + height

                            window_latent_left = latent_pos_left_start + shift_w_idx * latent_width
                            window_latent_right = window_latent_left + latent_width
                            window_latent_top = latent_pos_top_start + shift_h_idx * latent_height
                            window_latent_down = window_latent_top + latent_height

                            window_latent_frame_begin = latent_frames_begin + shift_f_idx * frames
                            window_latent_frame_end = window_latent_frame_begin + frames

                            if dock_at_h:
                                if shift_h_idx == -100: # dock at up edge
                                    if i % loop_step == 0:
                                        print(f"i % loop_step = {i} % {loop_step} = 0, no need for docking, skipped")
                                        continue
                                    window_latent_top = 0
                                    window_latent_down = window_latent_top + latent_height
                                    window_image_top = 0
                                    window_image_down = window_image_top + height

                                if shift_h_idx == -101: # dock at down edge
                                    if i % loop_step == 0:
                                        print(f"i % loop_step = {i} % {loop_step} = 0, no need for docking, skipped")
                                        continue
                                    window_latent_top = height * num_windows_h // self.vae_scale_factor - latent_height
                                    window_latent_down = window_latent_top + latent_height
                                    window_image_top = height * num_windows_h - height
                                    window_image_down = window_image_top + height

                                if window_latent_down > height * num_windows_h // self.vae_scale_factor:
                                    print(f"window_latent_down = {window_latent_down} > down edge = {height * num_windows_h // self.vae_scale_factor}, skipped because docking H")
                                    continue


                            window_latent = panorama_ring_latent_handler.get_window_latent(pos_left=window_latent_left,
                                                                                           pos_right=window_latent_right,
                                                                                           pos_top=window_latent_top,
                                                                                           pos_down=window_latent_down,
                                                                                           frame_begin=window_latent_frame_begin,
                                                                                           frame_end=window_latent_frame_end)

                            img_emb = panorama_ring_image_tensor_handler.get_encoded_image_cond(pretrained_t2v=self.pretrained_t2v,
                                                                                                pos_left=window_image_left,
                                                                                                pos_right=window_image_right,
                                                                                                pos_top=window_image_top,
                                                                                                pos_down=window_image_down)

                            window_denoised_mask = panorama_ring_mask_handler.get_window_latent(pos_left=window_latent_left,
                                                                                                pos_right=window_latent_right,
                                                                                                pos_top=window_latent_top,
                                                                                                pos_down=window_latent_down,
                                                                                                frame_begin=window_latent_frame_begin,
                                                                                                frame_end=window_latent_frame_end)

                            if merge_renoised_overlap_latent_ratio is not None and i < total_steps - 1:

                                noised_window_latent = self.scheduler.re_noise(x_a=window_latent.clone(),
                                                                               step_a=total_steps - i - 1 - 1,
                                                                               step_b=total_steps - i - 1)
                                window_denoised_mask = window_denoised_mask[0, 0, [0]]
                                window_latent = mix_latents_with_mask(latent_1=window_latent,
                                                                      latent_to_add=noised_window_latent,
                                                                      mask=window_denoised_mask,
                                                                      mix_ratio=merge_renoised_overlap_latent_ratio)



                            imtext_cond = torch.cat([text_emb, img_emb], dim=1)
                            cond = {"c_crossattn": [imtext_cond], "fps": fps}

                            print(f"window_idx: [{shift_f_idx}, {shift_h_idx}, {shift_w_idx}] (f, h, w) | \t "
                                  f"window_latent: f[{window_latent_frame_begin} - {window_latent_frame_end}] h[{window_latent_top} - {window_latent_down}] w[{window_latent_left} - {window_latent_right}] | \t"
                                  f"window image: h[{window_image_top} - {window_image_down}] w[{window_image_left} - {window_image_right}]")

                            kwargs.update({"clean_cond": True})

                            ts = torch.full((bs,), t, device=device, dtype=torch.long)  # [1]

                            model_pred_cond = self.pretrained_t2v.model( 
                                window_latent,
                                ts,
                                **cond,
                                curr_time_steps=ts,
                                temporal_length=frames,
                                **kwargs
                            )

                            if guidance_scale != 1.0:

                                model_pred_uncond = self.pretrained_t2v.model(
                                    window_latent,
                                    ts,
                                    **uncond,
                                    curr_time_steps=ts,
                                    temporal_length=frames,
                                    **kwargs
                                )

                                model_pred = model_pred_uncond + guidance_scale * (model_pred_cond - model_pred_uncond)

                            else:
                                model_pred = model_pred_cond

                            index = total_steps - i - 1

                            window_latent, denoised = self.scheduler.ddim_step(sample=window_latent, noise_pred=model_pred,
                                                                               indices=[index] * window_latent.shape[2])

                            panorama_ring_latent_handler.set_window_latent(window_latent,
                                                                           pos_left=window_latent_left,
                                                                           pos_right=window_latent_right,
                                                                           pos_top=window_latent_top,
                                                                           pos_down=window_latent_down,
                                                                           frame_begin=window_latent_frame_begin,
                                                                           frame_end=window_latent_frame_end)

                            panorama_ring_latent_denoised_handler.set_window_latent(denoised,
                                                                                    pos_left=window_latent_left,
                                                                                    pos_right=window_latent_right,
                                                                                    pos_top=window_latent_top,
                                                                                    pos_down=window_latent_down,
                                                                                    frame_begin=window_latent_frame_begin,
                                                                                    frame_end=window_latent_frame_end)

                            new_window_denoised_mask = torch.ones_like(window_latent, dtype=window_latent.dtype, device=window_latent.device)
                            panorama_ring_mask_handler.set_window_latent(new_window_denoised_mask,
                                                                         pos_left=window_latent_left,
                                                                         pos_right=window_latent_right,
                                                                         pos_top=window_latent_top,
                                                                         pos_down=window_latent_down,
                                                                         frame_begin=window_latent_frame_begin,
                                                                         frame_end=window_latent_frame_end)


                progress_bar.update()

        denoised = panorama_ring_latent_denoised_handler.torch_latent.clone().to(device=init_panorama_latent.device)

        if not output_type == "latent":
            videos = self.pretrained_t2v.decode_first_stage_2DAE(denoised) 
        else:
            videos = denoised

        return videos, denoised

    @torch.no_grad()
    def _add_noise(self, clear_video_latent, time_step_index):  
        ddim_alphas = self.scheduler.ddim_alphas
        alpha = ddim_alphas[time_step_index]
        beta = 1 - alpha
        noised_latent = (alpha ** 0.5) * clear_video_latent + (beta ** 0.5) * torch.randn_like(clear_video_latent)
        return noised_latent

    def tensor2image(self, batch_tensors):
        img_tensor = torch.squeeze(batch_tensors)  # c,h,w

        image = img_tensor.detach().cpu()
        image = torch.clamp(image.float(), -1., 1.)

        image = (image + 1.0) / 2.0
        image = (image * 255).to(torch.uint8).permute(1, 2, 0)  # h,w,c
        image = image.numpy()
        image = Image.fromarray(image)

        return image

    @torch.no_grad()
    def encode_image_cond(self, img_path, height, width):
        cond_images = self._load_imgs_from_paths(img_path_list=img_path, height=height, width=width)
        cond_images = cond_images.to(self.pretrained_t2v.device)
        img_emb = self.pretrained_t2v.get_image_embeds(cond_images)
        return img_emb
