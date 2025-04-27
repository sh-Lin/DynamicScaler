import os
import random
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
from lvdm.models.ddpm3d import LatentDiffusion
from pipeline.scheduler import lvdm_DDIM_Scheduler
from utils.diffusion_utils import resize_video_latent
from utils.shift_window_utils import RingLatent


logger = logging.get_logger(__name__)


class VC2_Pipeline_T2V(DiffusionPipeline): 

    scheduler: lvdm_DDIM_Scheduler
    pretrained_t2v: LatentDiffusion

    def __init__(
            self,
            pretrained_t2v: LatentDiffusion,
            scheduler: lvdm_DDIM_Scheduler,
            model_config: Dict[str, Any] = None,
    ):
        super().__init__()


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
                img_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1)
            else:
                print(f'ERROR: <{ext}> image loading only support format: [mp4], [png], [jpg]')
                raise NotImplementedError
            img_tensor = (img_tensor / 255. - 0.5) * 2
            batch_tensor.append(img_tensor)
        return torch.stack(batch_tensor, dim=0)


    @torch.no_grad()
    def basic_sample(
            self,
            prompt: Union[str, List[str]] = None,
            height: Optional[int] = 320,
            width: Optional[int] = 512,
            frames: int = 16,
            fps: int = 16,
            guidance_scale: float = 7.5,
            num_videos_per_prompt: Optional[int] = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
            latents: Optional[torch.FloatTensor] = None,
            num_inference_steps: int = 4,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",
            skip_time_step_idx=None,
            **kwargs
    ):

        unet_config = self.model_config["params"]["unet_config"]
        # 0. Default height and width to unet
        frames = self.pretrained_t2v.temporal_length if frames < 0 else frames

        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
            prompt = [prompt]
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        text_emb = self.pretrained_t2v.get_learned_conditioning(prompt)
        cond = {"c_crossattn": [text_emb], "fps": fps}



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

            if isinstance(cond, dict):
                uncond = {key: cond[key] for key in cond.keys()}
                uncond.update({'c_crossattn': [uc_emb]})
            else:
                uncond = uc_emb
        else:
            uncond = None


        # 4. Prepare timesteps
        self.scheduler.make_schedule(num_inference_steps)
        timesteps = np.flip(self.scheduler.ddim_timesteps)


        if skip_time_step_idx is not None:
            timesteps = timesteps[skip_time_step_idx:]
            print(f"skip : {skip_time_step_idx}")

        print(f"[basic_sample] denoise timesteps: {timesteps}")

        total_steps = self.scheduler.ddim_timesteps.shape[0]

        # 5. Prepare latent variable
        if latents is None:
            print("[basic_sample] latent is None, use full random init latent instead")
            assert (skip_time_step_idx is None) or (skip_time_step_idx == 0), "[basic_sample] skip time step should only work with prepared non full noise latents"
            num_channels_latents = unet_config["params"]["in_channels"]
            height = height // self.vae_scale_factor
            width = width // self.vae_scale_factor
            total_shape = (
                1,
                batch_size,
                num_channels_latents,
                frames,
                height,
                width,
            )
            print('total_shape', total_shape)
            latents = torch.randn(total_shape, device=device).repeat(1, batch_size, 1, 1, 1, 1)
            latents = latents[0]
        else:
            print("[basic_sample] using given init latent")

        bs = batch_size * num_videos_per_prompt # ?

        # 6. DDIM Sampling Loop
        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):

                kwargs.update({"clean_cond": True})

                ts = torch.full((bs,), t, device=device, dtype=torch.long) 
                model_pred_cond = self.pretrained_t2v.model(
                    latents,
                    ts,
                    **cond,
                    curr_time_steps=ts,
                    temporal_length=frames,
                    **kwargs
                )

                if guidance_scale != 1.0:

                    model_pred_uncond = self.pretrained_t2v.model(
                        latents,
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

                latents, denoised = self.scheduler.ddim_step(sample=latents, noise_pred=model_pred, indices=[index]*latents.shape[2])

                progress_bar.update()

        if not output_type == "latent":
            videos = self.pretrained_t2v.decode_first_stage_2DAE(denoised)
        else:
            videos = denoised

        return videos, denoised


    @torch.no_grad()
    def basic_sample_shift_multi_windows(
            self,
            prompt: Union[str, List[str]] = None,
            # img_cond_path: Union[str, List[str]] = None,
            height: Optional[int] = 320,
            width: Optional[int] = 512,
            frames: int = 16,
            fps: int = 16,
            guidance_scale: float = 7.5,
            num_videos_per_prompt: Optional[int] = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,  
            init_panorama_latent: torch.Tensor = None,
            clear_pre_denoised_latent: torch.Tensor = None,
            clear_pre_denoised_video_tensor: torch.Tensor = None,
            num_windows_w: int = None,                  
            num_windows_h: int = None,                  
            num_windows_f: int = None,
            loop_step: int = None,       
            latents: Optional[torch.FloatTensor] = None,
            num_inference_steps: int = 50,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",

            use_pre_denoise=False,
            pre_denoise_steps=None,
            skip_steps_after_pre_denoise=0,
            shift_jump_odd_w=False,
            shift_jump_odd_h=False,
            shift_jump_odd_f=False,
            docking_w=False,
            docking_h=False,
            docking_f=False,
            docking_step_range=None,
            merge_predenoise_ratio_list=None,   
            random_shuffle_init_frame_stride=0,
            sparse_add_residual=True,
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
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Encode input prompt
        text_emb = self.pretrained_t2v.get_learned_conditioning(prompt)
        cond = {"c_crossattn": [text_emb], "fps": fps}


        # 3.5 Prepare CFG if used
        if guidance_scale != 1.0:
            uncond_type = self.pretrained_t2v.uncond_type
            if uncond_type == "empty_seq":
                prompts = batch_size * [""]
                uc_emb = self.pretrained_t2v.get_learned_conditioning(prompts)
            elif uncond_type == "zero_embed":
                c_emb = cond["c_crossattn"][0] if isinstance(cond, dict) else cond
                uc_emb = torch.zeros_like(c_emb)
            else:
                raise NotImplementedError()

            if isinstance(cond, dict):
                uncond = {key: cond[key] for key in cond.keys()}
                uncond.update({'c_crossattn': [uc_emb]})
            else:
                uncond = uc_emb
        else:
            uncond = None

        # 4. Prepare timesteps
        self.scheduler.make_schedule(num_inference_steps)
        full_timesteps = np.flip(self.scheduler.ddim_timesteps)

        if use_skip_time and not progressive_skip:
            timesteps = full_timesteps[skip_time_step_idx-skip_steps_after_pre_denoise:]
            print(f"skip : {skip_time_step_idx}")
        else:
            timesteps = full_timesteps

        print(f"[basic_sample_shift_multi_windows] denoise timesteps: {timesteps}")
        print(f"[basic_sample_shift_multi_windows] SKIP {skip_time_step_idx}-{skip_steps_after_pre_denoise} = {skip_time_step_idx-skip_steps_after_pre_denoise} timesteps {'(progressive)' if progressive_skip else ''}")
        print(f"[basic_sample_shift_multi_windows] skip_steps_after_pre_denoise = {skip_steps_after_pre_denoise}")

        total_steps = len(timesteps)

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
        bs = batch_size * num_videos_per_prompt
        if init_panorama_latent is None:

            init_panorama_latent = torch.randn(total_shape, device=device).repeat(batch_size, 1, 1, 1, 1)


            if random_shuffle_init_frame_stride > 0:

                print(f"[basic_sample_shift_multi_windows] random_shuffle_init_frame_stride {random_shuffle_init_frame_stride}")

                for frame_index in range(frames, frames * num_windows_f, random_shuffle_init_frame_stride):
                    start_idx = frame_index - frames
                    end_idx = frame_index + random_shuffle_init_frame_stride - frames
                    list_index = list(range(start_idx, end_idx))
                    random.shuffle(list_index)
                    init_panorama_latent[:, :, :, frame_index:frame_index + random_shuffle_init_frame_stride] = init_panorama_latent[:, :, :, list_index]


            if use_skip_time:
                assert use_pre_denoise and pre_denoise_steps > 0, "[basic_sample_shift_multi_windows] skip ts should be used " \
                                                                  "with pre denoise if init_panorama_latent is not provided "
                assert skip_time_step_idx >= skip_steps_after_pre_denoise, f"[basic_sample_shift_multi_windows] skip_time_step_idx {skip_time_step_idx} should >=" \
                                                                           f"skip_steps_after_pre_denoise {skip_steps_after_pre_denoise}"



            if use_pre_denoise and pre_denoise_steps > 0:

                if (num_windows_h != 1 or num_windows_w != 1) and num_windows_f != 1:
                    raise NotImplementedError()
                _basic_latent_shape = (
                    batch_size,
                    num_channels_latents,
                    frames,
                    latent_height,
                    latent_width,
                )
                latent = torch.randn(_basic_latent_shape, device=device).repeat(batch_size, 1, 1, 1, 1)

                resized_latent = None

                if clear_pre_denoised_video_tensor is not None:
                    resized_video_tensor = resize_video_latent(input_latent=clear_pre_denoised_video_tensor.clone(), mode="bicubic",
                                                               target_height=height * num_windows_h,
                                                               target_width=width * num_windows_w)
                    resized_latent = self.pretrained_t2v.encode_first_stage_2DAE(resized_video_tensor).clone()
                    assert list(resized_latent.shape) == list(total_shape)

                elif clear_pre_denoised_latent is not None:
                    assert list(clear_pre_denoised_latent.shape) == list(_basic_latent_shape), f"[basic_sample_shift_multi_windows] " \
                                                                                               f"clear_pre_denoised_latent shape :{clear_pre_denoised_latent}" \
                                                                                               f"not equal to _basic_latent_shape: {_basic_latent_shape}"
                    latent = clear_pre_denoised_latent.clone()
                    resized_latent = resize_video_latent(input_latent=latent.clone(), mode="bicubic",
                                                         target_height=latent_height * num_windows_h,
                                                         target_width=latent_width * num_windows_w)
                else:
                    print(f"[basic_sample_shift_multi_windows] Pre Denosing {pre_denoise_steps} Steps...")
                    for i, t in enumerate(full_timesteps[:pre_denoise_steps]):
                        latent, denoised = self._basic_denoise_one_step(t=t, i=i, total_steps=total_steps,
                                                                        device=device, latent=latent,
                                                                        cond=cond, uncond=uncond, guidance_scale=guidance_scale,
                                                                        frames=frames, bs=bs)
                    resized_latent = resize_video_latent(input_latent=latent.clone(), mode="bicubic",
                                                         target_height=latent_height * num_windows_h,
                                                         target_width=latent_width * num_windows_w)

                init_panorama_latent = self._add_noise(clear_video_latent=resized_latent, time_step_index=total_steps-1)

                if use_skip_time:

                    if progressive_skip:

                        for frame_idx, progs_skip_idx in enumerate(list(reversed(range(skip_time_step_idx)))):

                            noised_frame_latent = self._add_noise(clear_video_latent=resized_latent[:, :, [frame_idx]],
                                                                  time_step_index=total_steps-progs_skip_idx-1)
                            init_panorama_latent[:, :, [frame_idx]] = noised_frame_latent.clone()

                    else:
                        init_panorama_latent = self._add_noise(clear_video_latent=resized_latent, time_step_index=total_steps-1)


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
        # 6. DDIM Sampling Loop
        with self.progress_bar(total=len(timesteps)) as progress_bar:

            for i, t in enumerate(timesteps):

                latent_pos_left_start = (i % loop_step) * latent_step_size_w
                latent_pos_top_start = (i % loop_step) * latent_step_size_h
                latent_frames_begin = (i % loop_step) * latent_step_size_f

                if use_pre_denoise and merge_predenoise_ratio_list is not None and resized_latent is not None:

                    assert len(merge_predenoise_ratio_list) == len(timesteps), f"merge_predenoise_ratio_list " \
                                                                               f"({len(merge_predenoise_ratio_list)}) " \
                                                                               f"should have same length as timesteps" \
                                                                               f"({len(timesteps)})"


                    curr_merge_ratio = merge_predenoise_ratio_list[i]
                    print(f"merging residual latent: {round(curr_merge_ratio, 3)} * curr + {round(1.0-curr_merge_ratio, 3)} * noised_resized")

                    curr_latent = panorama_ring_latent_handler.torch_latent.clone()
                    noised_resized_latent = self.scheduler.re_noise(x_a=resized_latent.clone(),
                                                                    step_a=0,
                                                                    step_b=total_steps - i - 1
                                                                    )

                    if sparse_add_residual:
                        mixed_residual_latent = curr_latent.clone()
                        mixed_residual_latent[..., i%2::2, ::2] = curr_merge_ratio * curr_latent[..., (i+1)%2::2, ::2] + (1.0 - curr_merge_ratio) * noised_resized_latent[..., ::2, ::2]
                        mixed_residual_latent[..., (i+1)%2::2, 1::2] = curr_merge_ratio * curr_latent[..., i%2::2, 1::2] + (1.0 - curr_merge_ratio) * noised_resized_latent[..., ::2, ::2]
                    else:
                        mixed_residual_latent = curr_latent * curr_merge_ratio + noised_resized_latent * (1.0 - curr_merge_ratio)
                    panorama_ring_latent_handler.torch_latent = mixed_residual_latent.clone()


                if i % 2 == 1 and shift_jump_odd_h and num_windows_h > 1:
                    latent_pos_left_start = latent_pos_left_start + (latent_width * num_windows_w // 2)
                if i % 2 == 1 and shift_jump_odd_w and num_windows_w > 1:
                    latent_pos_top_start = latent_pos_top_start + (latent_height * num_windows_h // 2)
                if i % 2 == 1 and shift_jump_odd_f and num_windows_f > 1:
                    latent_frames_begin = latent_frames_begin + (frames * num_windows_f // 2)

                print(f"\n"
                      f"i = {i}, t = {t}")

                for shift_f_idx in (range(-1, num_windows_f) if docking_f else range(num_windows_f)):

                    for shift_w_idx in (range(-1, num_windows_w) if docking_w else range(num_windows_w)):

                        for shift_h_idx in (range(-1, num_windows_h) if docking_h else range(num_windows_h)):

                            window_latent_left = latent_pos_left_start + shift_w_idx * latent_width
                            window_latent_right = window_latent_left + latent_width
                            window_latent_top = latent_pos_top_start + shift_h_idx * latent_height
                            window_latent_down = window_latent_top + latent_height
                            window_latent_frame_begin = latent_frames_begin + shift_f_idx * frames
                            window_latent_frame_end = window_latent_frame_begin + frames

                            if docking_w and i in docking_step_range:
                                if shift_w_idx == -1:
                                    window_latent_left = 0
                                    window_latent_right = latent_width
                                if shift_w_idx == num_windows_w - 1:
                                    window_latent_left = latent_width * (num_windows_w-1)
                                    window_latent_right = latent_width * num_windows_w
                            elif shift_w_idx == -1:
                                continue

                            if docking_h and i in docking_step_range:
                                if shift_h_idx == -1:
                                    window_latent_top = 0
                                    window_latent_down = latent_height
                                if shift_h_idx == num_windows_h - 1:
                                    window_latent_top = latent_height * (num_windows_h-1)
                                    window_latent_down = latent_height * num_windows_h
                            elif shift_h_idx == -1:
                                continue

                            if docking_f and i in docking_step_range:
                                if shift_f_idx == -1:
                                    window_latent_frame_begin = 0
                                    window_latent_frame_end = frames
                                if shift_f_idx == num_windows_f - 1:
                                    window_latent_frame_begin = frames * (num_windows_f-1)
                                    window_latent_frame_end = frames * num_windows_f
                            elif shift_f_idx == -1:
                                continue



                            window_latent = panorama_ring_latent_handler.get_window_latent(pos_left=window_latent_left,
                                                                                           pos_right=window_latent_right,
                                                                                           pos_top=window_latent_top,
                                                                                           pos_down=window_latent_down,
                                                                                           frame_begin=window_latent_frame_begin,
                                                                                           frame_end=window_latent_frame_end)

                            print(f"window_idx: [{shift_f_idx}, {shift_h_idx}, {shift_w_idx}] (f, h, w) | \t "
                                  f"window_latent: f[{window_latent_frame_begin} - {window_latent_frame_end}] h[{window_latent_top} - {window_latent_down}] w[{window_latent_left} - {window_latent_right}] | \t"
                                  f"")

                            window_latent, denoised = self._basic_denoise_one_step(t=t, i=i, total_steps=total_steps,
                                                                                   device=device, latent=window_latent,
                                                                                   cond=cond, uncond=uncond,
                                                                                   guidance_scale=guidance_scale,
                                                                                   frames=frames, bs=bs)

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

                progress_bar.update()

        denoised = panorama_ring_latent_denoised_handler.torch_latent.clone().to(device=init_panorama_latent.device)

        if not output_type == "latent":
            videos = self.pretrained_t2v.decode_first_stage_2DAE(denoised)
        else:
            videos = denoised

        return videos, denoised


    @torch.no_grad()
    def _basic_denoise_one_step(self,
                                t, i, total_steps,
                                device,
                                latent,
                                cond, uncond, guidance_scale,
                                frames,
                                bs=1,
                                **kwargs
                                ):
        kwargs.update({"clean_cond": True})

        ts = torch.full((bs,), t, device=device, dtype=torch.long)

        model_pred_cond = self.pretrained_t2v.model(
            latent,
            ts,
            **cond,
            curr_time_steps=ts,
            temporal_length=frames,
            **kwargs
        )

        if guidance_scale != 1.0:

            model_pred_uncond = self.pretrained_t2v.model(
                latent,
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

        latent, denoised = self.scheduler.ddim_step(sample=latent, noise_pred=model_pred,
                                                    indices=[index] * latent.shape[2])

        return latent, denoised


    @torch.no_grad()
    def _add_noise(self, clear_video_latent, time_step_index):
        clear_video_latent = clear_video_latent.clone()
        ddim_alphas = self.scheduler.ddim_alphas
        alpha = ddim_alphas[time_step_index]
        beta = 1 - alpha
        noised_latent = (alpha ** 0.5) * clear_video_latent + (beta ** 0.5) * torch.randn_like(clear_video_latent)
        return noised_latent



