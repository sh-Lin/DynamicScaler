import numpy as np

import torch
from PIL import Image

from tqdm import trange
from typing import List, Optional, Union, Dict, Any
from diffusers import logging

from pipeline.t2v_normal_pipeline import VC2_Pipeline_T2V
from utils.diffusion_utils import resize_video_latent
from utils.multi_prompt_utils import select_prompt_from_multi_prompt_dict_by_factor
from utils.shift_window_utils import RingLatent, RingImageTensor
from utils.tensor_utils import load_image_tensor_from_path, mix_latents_with_mask

from utils.precast_latent_utils import encode_images_list_to_latent_tensor
from utils.panorama_tensor_utils import PanoramaTensor, PanoramaLatentProxy


class VC2_Pipeline_T2V_SpherePano(VC2_Pipeline_T2V):


    @torch.no_grad()
    def basic_sample_shift_shpere_panorama(
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
            init_sphere_latent: torch.Tensor = None, 

            equirect_width: int = None,
            equirect_height: int = None,

            phi_theta_dict: dict = None,            
            phi_prompt_dict: dict = None,           

            view_fov: int = None,

            view_get_scale_factor: int = 1,        
            view_set_scale_factor: int = 1,        
            loop_step_theta: int = None,            
            merge_renoised_overlap_latent_ratio: float = None,                                                                                   
            phi_fov_dict: dict = None, 
            denoise_to_step: int = None,  

            latents: Optional[torch.FloatTensor] = None,
            num_inference_steps: int = 4,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",

            downsample_factor_before_vae_decode=None,
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

        self.scheduler.make_schedule(num_inference_steps)
        timesteps = np.flip(self.scheduler.ddim_timesteps)  # [ 999, ... , 0 ]


        if use_skip_time and not progressive_skip:
            timesteps = timesteps[skip_time_step_idx:]
            print(f"skip : {skip_time_step_idx}")

        if denoise_to_step is not None:
            assert not (use_skip_time and not progressive_skip), 'should not use denoise_to_step while using Non progressive time step skip. ' \
                                                                 'which may lead to disordered time step count in following steps'
            if use_skip_time and not progressive_skip:
                assert skip_time_step_idx < denoise_to_step, f'denoise_to_step {denoise_to_step} should not stop denoise earlier than progressive skip {skip_time_step_idx}, ' \
                                                             'you may have set undesired values'

            timesteps = timesteps[:denoise_to_step]
            print(f"Denoised to : {denoise_to_step}")

        print(f"[basic_sample_shift_multi_windows] denoise timesteps: {timesteps}")
        if use_skip_time:
            print(f"[basic_sample_shift_multi_windows] SKIP {skip_time_step_idx} timesteps {'(progressive)' if progressive_skip else ''}")

        total_steps = self.scheduler.ddim_timesteps.shape[0]

        # 5. Prepare latent variable [pano]
        num_channels_latents = unet_config["params"]["in_channels"]
        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor

        sphere_latent_shape = (
            batch_size,
            num_channels_latents,
            frames, # * num_windows_f,
            equirect_height // self.vae_scale_factor, # latent_height * num_windows_h,
            equirect_width // self.vae_scale_factor, # latent_width * num_windows_w,
        )

        if init_sphere_latent is None:

            init_sphere_latent = torch.randn(sphere_latent_shape, device=device).repeat(batch_size, 1, 1, 1, 1)

            if use_skip_time:

                raise NotImplementedError       # TODO

        else:
            print("[basic_sample_shift_multi_windows] using given init latent")
            assert init_sphere_latent.shape == sphere_latent_shape, f"[basic_sample_shift_multi_windows] " \
                                                                      f"init_panorama_latent shape {init_sphere_latent.shape}" \
                                                                      f"does not match" \
                                                                      f"desired shape {init_sphere_latent}"
            init_sphere_latent = init_sphere_latent.clone()


        panorama_sphere_latent_handler = PanoramaLatentProxy(equirect_tensor=init_sphere_latent)
        panorama_sphere_denoised_latent_handler = PanoramaLatentProxy(equirect_tensor=torch.zeros_like(init_sphere_latent))
        panorama_sphere_denoised_mask_handler = PanoramaTensor(equirect_tensor=torch.zeros_like(init_sphere_latent[0, 0, [0]]))

        bs = batch_size * num_videos_per_prompt

        # 6. DDIM Sampling Loop
        with self.progress_bar(total=len(timesteps)) as progress_bar:

            for i, t in enumerate(timesteps):

                phi_offset = 0 
                theta_offset = (i % loop_step_theta) * (view_fov // loop_step_theta)

                print(f"\n"
                      f"i = {i}, t = {t}")
                print(f"theta offset = {theta_offset}, phi_offset = {phi_offset}")

                # reset mask record
                panorama_sphere_denoised_mask_handler = PanoramaTensor(equirect_tensor=torch.zeros_like(init_sphere_latent[0, 0, [0]]))

                for phi_angle in list(phi_theta_dict.keys()):

                    theta_angles = phi_theta_dict[phi_angle]

                    # random.shuffle(theta_angles)

                    for theta_angle in theta_angles:

                        curr_phi_angle = phi_angle + phi_offset
                        curr_theta_angle = theta_angle + theta_offset

                        if phi_fov_dict is not None:
                            curr_fov = phi_fov_dict.get(curr_phi_angle, view_fov)
                        else:
                            curr_fov = view_fov

                        view_latent, _ = panorama_sphere_latent_handler.get_view_tensor_no_interpolate(fov=view_fov,
                                                                                                       theta=curr_theta_angle,
                                                                                                       phi=curr_phi_angle,
                                                                                                       width=latent_width * view_get_scale_factor,
                                                                                                       height=latent_height * view_get_scale_factor)

                        if view_get_scale_factor != 1:
                            view_latent = resize_video_latent(input_latent=view_latent, mode="nearest",
                                                              target_width=latent_width,
                                                              target_height=latent_height)

                        view_denoised_mask, _ = panorama_sphere_denoised_mask_handler.get_view_tensor_no_interpolate(
                            fov=curr_fov,
                            theta=curr_theta_angle,
                            phi=curr_phi_angle,
                            width=latent_width,
                            height=latent_height)   # [1, H, W], 0 为没去噪过的部分, 1为已经去噪过的部分

                        if merge_renoised_overlap_latent_ratio is not None and i < total_steps - 1 :

                            noised_view_latent = self.scheduler.re_noise(x_a=view_latent.clone(),
                                                                         step_a=total_steps - i - 1 - 1,
                                                                         step_b=total_steps - i - 1)

                            view_latent = mix_latents_with_mask(latent_1=view_latent,
                                                                latent_to_add=noised_view_latent,
                                                                mask=view_denoised_mask,
                                                                mix_ratio=merge_renoised_overlap_latent_ratio)

                        print_prompt = prompt
                        if phi_prompt_dict is not None:
                            curr_phi_prompt = phi_prompt_dict[phi_angle]
                            text_emb = self.pretrained_t2v.get_learned_conditioning([curr_phi_prompt])
                            print_prompt = curr_phi_prompt

                        imtext_cond = torch.cat([text_emb], dim=1)
                        cond = {"c_crossattn": [imtext_cond], "fps": fps}

                        print(f"window: phi = {curr_phi_angle}, theta = {curr_theta_angle}, prompt = {print_prompt}, fov = {curr_fov}")

                        kwargs.update({"clean_cond": True})

                        ts = torch.full((bs,), t, device=device, dtype=torch.long)

                        model_pred_cond = self.pretrained_t2v.model(
                            view_latent,
                            ts,
                            **cond,
                            curr_time_steps=ts,
                            temporal_length=frames,
                            **kwargs
                        )

                        if guidance_scale != 1.0:

                            model_pred_uncond = self.pretrained_t2v.model(
                                view_latent,
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

                        view_latent, denoised = self.scheduler.ddim_step(sample=view_latent, noise_pred=model_pred,
                                                                         indices=[index] * view_latent.shape[2])

                        if view_set_scale_factor != 1:
                            view_latent = resize_video_latent(input_latent=view_latent, mode="nearest",
                                                              target_width=latent_width * view_set_scale_factor,
                                                              target_height=latent_height * view_set_scale_factor)
                            denoised = resize_video_latent(input_latent=denoised, mode="nearest",
                                                           target_width=latent_width * view_set_scale_factor,
                                                           target_height=latent_height * view_set_scale_factor)

                        panorama_sphere_latent_handler.set_view_tensor_no_interpolation(
                            view_tensor=view_latent,
                            fov=curr_fov,
                            theta=curr_theta_angle,
                            phi=curr_phi_angle)

                        panorama_sphere_denoised_latent_handler.set_view_tensor_no_interpolation(view_tensor=denoised,
                                                                                                 fov=curr_fov,
                                                                                                 theta=curr_theta_angle,
                                                                                                 phi=curr_phi_angle)

                        new_view_denoised_mask = torch.ones_like(view_latent[0, 0, [0]], dtype=view_latent.dtype, device=view_latent.device)
                        panorama_sphere_denoised_mask_handler.set_view_tensor_no_interpolation(view_tensor=new_view_denoised_mask,
                                                                                               fov=curr_fov,
                                                                                               theta=curr_theta_angle,
                                                                                               phi=curr_phi_angle)

                progress_bar.update()

        denoised = panorama_sphere_denoised_latent_handler.get_equirect_tensor().to(device=self.pretrained_t2v.device)
        final_latents = panorama_sphere_latent_handler.get_equirect_tensor().to(device=self.pretrained_t2v.device)

        if downsample_factor_before_vae_decode is not None:
            B, C, N, H, W = denoised.shape
            denoised = resize_video_latent(input_latent=denoised.clone(), mode="nearest",
                                           target_height=int(H // downsample_factor_before_vae_decode),
                                           target_width=int(W // downsample_factor_before_vae_decode))
            final_latents = resize_video_latent(input_latent=final_latents.clone(), mode="nearest",
                                                target_height=int(H // downsample_factor_before_vae_decode),
                                                target_width=int(W // downsample_factor_before_vae_decode))

        if not output_type == "latent":
            videos = self.pretrained_t2v.decode_first_stage_2DAE(denoised)
        else:
            videos = final_latents

        return videos, denoised



    @torch.no_grad()
    def basic_sample_shift_multi_windows(
            self,
            prompt: Union[str, List[str]] = None,

            height: Optional[int] = 320,
            width: Optional[int] = 512,
            frames: int = 16,
            fps: int = 16,
            guidance_scale: float = 7.5,
            num_videos_per_prompt: Optional[int] = 1,
            generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,  
            init_panorama_latent: torch.Tensor = None,  
            total_w: int = None,
            total_h: int = None,                        
            num_windows_w: int = None,                  
            num_windows_h: int = None,
            num_windows_f: int = None,                  
            loop_step: int = None,                      
            dock_at_h = None,
            latents: Optional[torch.FloatTensor] = None,
            num_inference_steps: int = 4,
            prompt_embeds: Optional[torch.FloatTensor] = None,
            output_type: Optional[str] = "pil",

            merge_renoised_overlap_latent_ratio:float = 1,

            window_multi_prompt_dict: dict = None,

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

        if use_skip_time and not progressive_skip:
            timesteps = timesteps[skip_time_step_idx:]
            print(f"skip : {skip_time_step_idx}")


        print(f"[basic_sample_shift_multi_windows] denoise timesteps: {timesteps}")
        print(f"[basic_sample_shift_multi_windows] SKIP {skip_time_step_idx} timesteps {'(progressive)' if progressive_skip else ''}")

        total_steps = len(timesteps)
        # 5. Prepare latent variable [pano]
        num_channels_latents = unet_config["params"]["in_channels"]
        latent_height = height // self.vae_scale_factor
        latent_width = width // self.vae_scale_factor
        total_shape = (
            batch_size,
            num_channels_latents,
            frames * num_windows_f,
            total_h // self.vae_scale_factor,
            total_w // self.vae_scale_factor,
        )

        if init_panorama_latent is None:

            init_panorama_latent = torch.randn(total_shape, device=device).repeat(batch_size, 1, 1, 1, 1)

            if use_skip_time:

                raise NotImplementedError   # TODO

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
        overlap_ratio_w = 1 - (total_w/width - 1) / (num_windows_w - 1)

        image_window_step_size_w = int(width * (1 - overlap_ratio_w))       # TODO :CHECK
        latent_window_step_size_w = image_window_step_size_w // self.vae_scale_factor

        image_offset_step_size_w = int((1 - overlap_ratio_w) * width / loop_step)
        latent_offset_step_size_w = image_offset_step_size_w // self.vae_scale_factor
        if num_windows_w == 1:
            image_offset_step_size_w = 0
            latent_offset_step_size_w = 0
        print(f"Shift for W: \n"
              f"overlape_ratio_w = 1 - (total_w/width - 1) / (num_windows_w - 1) = 1 - ({total_w}/{width} - 1) / ({num_windows_w} - 1) = {overlap_ratio_w}\n"
              f"image_window_step_size_w = int(width * (1 - overlape_ratio_w)) = int({width} * (1 - {overlap_ratio_w})) = int({width * (1 - overlap_ratio_w)}) = {image_window_step_size_w}\n"
              f"image_offset_step_size_w = int(overlap_ratio_w * total_w / offset_loop_step) = int({overlap_ratio_w} * {total_w} / {loop_step}) = int({overlap_ratio_w * total_w / loop_step}) = {image_offset_step_size_w}")
        assert 0 <= overlap_ratio_w < 1, "overlap ratio for W is not legal"
        assert latent_offset_step_size_w, "latent_offset_step_size_w <= 0 ! consider increase W windows"

        overlap_ratio_h = 1 - (total_h/height - 1) / (num_windows_h - 1)

        image_window_step_size_h = int(height * (1 - overlap_ratio_h))
        latent_window_step_size_h = image_window_step_size_h // self.vae_scale_factor

        image_offset_step_size_h = int((1 - overlap_ratio_h) * height / loop_step)
        latent_offset_step_size_h = image_offset_step_size_h // self.vae_scale_factor
        if num_windows_h == 1:
            image_offset_step_size_h = 0
            latent_offset_step_size_h = 0
        print(f"Shift for H: \n"
              f"overlape_ratio_h = 1 - (total_h/height - 1) / (num_windows_h - 1) = 1 - ({total_h}/{height} - 1) / ({num_windows_h} - 1) = {overlap_ratio_h}\n"
              f"image_window_step_size_h = int(height * (1 - overlape_ratio_h)) = int({height} * (1 - {overlap_ratio_h})) = int({width * (1 - overlap_ratio_h)}) = {image_window_step_size_h}\n"
              f"image_offset_step_size_h = int(overlap_ratio_h * total_h / offset_loop_step) = int({overlap_ratio_h} * {total_h} / {loop_step}) = int({overlap_ratio_h * total_h / loop_step}) = {image_offset_step_size_h}")
        assert 0 <= overlap_ratio_h < 1, "overlap ratio for H is not legal"
        assert latent_offset_step_size_h > 0, "latent_offset_step_size_h <= 0 ! consider increase H windows"

        latent_step_size_f = frames // loop_step
        if num_windows_f == 1:
            latent_step_size_f = 0

        assert latent_step_size_f > 0 or num_windows_f == 1, f"[basic_sample_shift_multi_windows] loop_step {loop_step} " \
                                                             f"> frames {frames} while num_windows_f {num_windows_f} > 0"
        bs = batch_size * num_videos_per_prompt

        # 6. DDIM Sampling Loop
        with self.progress_bar(total=len(timesteps)) as progress_bar:
            for i, t in enumerate(timesteps):

                image_pos_left_start = (i % loop_step) * image_offset_step_size_w       
                image_pos_top_start = (i % loop_step) * image_offset_step_size_h        

                latent_pos_left_start = (i % loop_step) * latent_offset_step_size_w    
                latent_pos_top_start = (i % loop_step) * latent_offset_step_size_h

                latent_frames_begin = (i % loop_step) * latent_step_size_f

                print(f"\n"
                      f"i = {i}, t = {t}")

                panorama_ring_mask_handler = RingLatent(init_latent=torch.zeros_like(init_panorama_latent))

                for shift_f_idx in range(num_windows_f):

                    for shift_w_idx in range(num_windows_w):

                        shift_h_idies_list = list(range(num_windows_h))
                        if dock_at_h:
                            shift_h_idies_list = [-100] + list(range(num_windows_h)) + [-101]

                        for shift_h_idx in shift_h_idies_list:

                            # TODO
                            window_latent_left = latent_pos_left_start + shift_w_idx * latent_window_step_size_w
                            window_latent_right = window_latent_left + latent_width
                            window_latent_top = latent_pos_top_start + shift_h_idx * latent_window_step_size_h
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

                                if shift_h_idx == -101: # dock at down edge
                                    if i % loop_step == 0:
                                        print(f"i % loop_step = {i} % {loop_step} = 0, no need for docking, skipped")
                                        continue
                                    window_latent_top = total_h // self.vae_scale_factor - latent_height
                                    window_latent_down = window_latent_top + latent_height

                                if window_latent_down > total_h // self.vae_scale_factor:
                                    print(f"window_latent_down = {window_latent_down} > down edge = {total_h // self.vae_scale_factor}, skipped because docking H")
                                    continue




                            window_latent = panorama_ring_latent_handler.get_window_latent(pos_left=window_latent_left,
                                                                                           pos_right=window_latent_right,
                                                                                           pos_top=window_latent_top,
                                                                                           pos_down=window_latent_down,
                                                                                           frame_begin=window_latent_frame_begin,
                                                                                           frame_end=window_latent_frame_end)
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

                            curr_h_factor = window_latent_down / (total_h//self.vae_scale_factor)
                            curr_prompt = prompt
                            if window_multi_prompt_dict is not None:
                                curr_prompt = select_prompt_from_multi_prompt_dict_by_factor(prompt_dict=window_multi_prompt_dict,
                                                                                             factor=curr_h_factor)
                                text_emb = self.pretrained_t2v.get_learned_conditioning([curr_prompt])
                            imtext_cond = torch.cat([text_emb], dim=1)
                            cond = {"c_crossattn": [imtext_cond], "fps": fps}

                            print(f"window_idx: [{shift_f_idx}, {shift_h_idx}, {shift_w_idx}] (f, h, w) | \t "
                                  f"window_latent: f[{window_latent_frame_begin} - {window_latent_frame_end}] h[{window_latent_top} - {window_latent_down}] w[{window_latent_left} - {window_latent_right}] | \t"
                                  f"curr prompt ({curr_h_factor}): {curr_prompt}"
                                  )

                            kwargs.update({"clean_cond": True})

                            ts = torch.full((bs,), t, device=device, dtype=torch.long)

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

            denoised_chunk_num = 16
            denoised_chunked = list(torch.chunk(denoised, denoised_chunk_num, dim=4))
            denoised_chunked_cat_list = [denoised_chunked[-1]] + denoised_chunked + [denoised_chunked[0]]
            denoised = torch.cat(denoised_chunked_cat_list, dim=4)

            video_frames_list = []

            for frame_idx in range(frames * num_windows_f):
                denoised_frame_latent = denoised[:, :, [frame_idx]]
                video_frames_tensor = self.pretrained_t2v.decode_first_stage_2DAE(denoised_frame_latent)
                video_frames_list.append(video_frames_tensor)

            videos = torch.cat(video_frames_list, dim=2)
            videos_chunked = torch.chunk(videos, denoised_chunk_num + 2, dim=4)

            videos = torch.cat(videos_chunked[1:-1], dim=4)

        else:
            videos = denoised

        return videos, denoised



