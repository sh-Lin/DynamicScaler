
import json
from datetime import datetime
import os
import shutil


import argparse
from dataclasses import dataclass, field

from dataclasses import dataclass
from typing import Dict


@dataclass
class SP_INPUT:
    prompt: str
    pano_image_path: str
    phi_prompt_dict: Dict[int, str]
    window_multi_prompt_dict: Dict[float, str] = None

@dataclass
class VArgs:

    # ============ CONFIGS ============= #
    seed:   int = 2333333
    gpu_id: int = 0

    sp_inputs = SP_INPUT(
        prompt="Massive green blue ocean wave, dynamic ocean spray, dynamic water motion, breaking waves with white foam, seabirds in motion, turquoise water transparency, distant ocean horizon",
        pano_image_path="./input/pano_surfing_1.png",
        phi_prompt_dict = {
            90: "Clear light blue sky",
            75: "Clear light blue sky",
            60: "Clear light blue sky",
            45: "Massive green blue ocean wave, dynamic ocean spray, dynamic water motion, breaking waves with white foam, seabirds in motion, turquoise water transparency, distant ocean horizon",
            0:  "Massive green blue ocean wave, dynamic ocean spray, dynamic water motion, breaking waves with white foam, seabirds in motion, turquoise water transparency, distant ocean horizon",
            -45: "green blue ocean with waves and swirling foam patterns",
            -60: "green blue ocean with waves",
            -75: "green blue ocean water",
            -90: "green blue ocean water",
        }
    )

    pano_image_path = sp_inputs.pano_image_path
    prompt = sp_inputs.prompt
    phi_prompt_dict = sp_inputs.phi_prompt_dict

    total_f: int = 16 
    do_upscale: bool = True 
    upscale_factor = 2

    # ============ ADVANCED CONFIGS ============= #
    phi_num:            int = 6     # 6
    view_fov: int           = 120
    denoise_to_step:    int = 15    # 5
    skip_time_step:     int = -1
    loop_step_theta:    int = 10
    predenoised_SP_latent_path: str = None 
    predenoised_SW_1x_latent_path: str = None
    dock_at_f: bool = True
    loop_step_frame: int = 8
    skip_1x = False
    loop_step_hw: int = 16
    merge_renoised_overlap_latent_ratio =  1 
    merge_denoised = True
    max_merge_denoised_overlap_latent_ratio = 0.5 
    _merge_prev_step = 20


    @classmethod
    def from_args(cls):
        parser = argparse.ArgumentParser()
        for field_name, field_def in cls.__dataclass_fields__.items():
            parser.add_argument(
                f'--{field_name}',
                type=type(field_def.default),
                default=field_def.default,
                help=f'{field_name} (default: {field_def.default})'
            )
        args = parser.parse_args()
        return cls(**vars(args))


vargs = VArgs.from_args()
print(vargs)

os.environ["CUDA_VISIBLE_DEVICES"] = f"{vargs.gpu_id}"
os.environ["WORLD_SIZE"] = "1"

from utils.loop_merge_utils import save_decoded_video_latents
from pipeline.i2v_sphere_panorama_pipeline import VC2_Pipeline_I2V_SpherePano
from pipeline.scheduler import lvdm_DDIM_Scheduler
from utils.precast_latent_utils import encode_images_list_to_latent_tensor, get_img_list_from_folder

from utils.diffusion_utils import resize_video_latent

import torch
import imageio

from dataclasses import dataclass, field
from typing import Optional
from collections import OrderedDict

from pytorch_lightning import seed_everything
from omegaconf import OmegaConf
from scripts.evaluation.funcs import load_model_checkpoint

from utils.utils import instantiate_from_config, create_dir

@dataclass
class RunArgs:
    config: str = "configs/inference_i2v_512_v1.0.yaml"
    base_ckpt_path: str = "./videocrafter_models/i2v_512_v1/model.ckpt"
    seed: int = 2333
    num_inference_steps: int = None
    total_video_length: int = 64
    num_processes: int = 1
    rank: int = 0
    height: int = 320
    width: int = 512
    save_frames: bool = True
    fps: int = 8
    unconditional_guidance_scale: float = 7.5
    lookahead_denoising: bool = False
    eta: float = 1.0
    output_dir: Optional[str] = None
    use_mp4: bool = True

def main(run_args: RunArgs, prompt, image_path, image_folder,
         use_fp16=False, save_latents=False,
         pano_image_path=None,
         loop_step=None,
         num_windows_h=None,
         num_windows_w=None,
         num_windows_f=None,
         use_skip_time=False,
         skip_time_step_idx=0,
         progressive_skip=False,
         equirect_width=None,
         equirect_height=None,
         phi_theta_dict=None,
         phi_prompt_dict: dict = None,
         view_fov=None,
         loop_step_theta=None,
         merge_renoised_overlap_latent_ratio=None,
         paste_on_static=None,
         downsample_factor_before_vae_decode=None,
         view_get_scale_factor=None,
         view_set_scale_factor=None,
         denoise_to_step=None,
         num_windows_h_2=None,
         num_windows_w_2=None,
         total_f=None,
         dock_at_f=None,
         loop_step_frame=None,
         overlap_ratio_list_1_f=None,
         overlap_ratio_list_2_f=None,
         upscale_factor=None,
         merge_prev_denoised_ratio_list=None,
         project_name="",
         project_folder=None):

    print(f"==========================\n"
          f"CURR GPU: {os.environ['CUDA_VISIBLE_DEVICES']}, SEED: {run_args.seed}\n"
          f"==========================\n")

    seed_everything(run_args.seed)

    output_dir, tmp_dir = create_dir(project_id=project_name, project_folder=project_folder)


    source_file_path = __file__
    destination_file_path = os.path.join(output_dir, "_src_script.py")
    shutil.copy(source_file_path, destination_file_path)

    src_path = os.path.join(output_dir, 'src')
    os.makedirs(src_path)

    src_dirs_list = ["./utils", "./pipeline"]

    for src_dir in src_dirs_list:
        src_dir_abs = os.path.abspath(src_dir)
        target_save_dir = os.path.join(src_path, src_dir)
        target_save_dir_abs = os.path.abspath(target_save_dir)
        shutil.copytree(src_dir_abs, target_save_dir_abs, ignore=shutil.ignore_patterns('*.pyc'))

    config = OmegaConf.load(run_args.config)
    model_config = config.pop("model", OmegaConf.create())

    if use_fp16: 
        model_config['params']['unet_config']['params']['use_fp16'] = True

    model = instantiate_from_config(model_config)
    model = model.cuda()
    assert os.path.exists(run_args.base_ckpt_path), f"Error: checkpoint [{run_args.base_ckpt_path}] Not Found!"

    model = load_model_checkpoint(model, run_args.base_ckpt_path)
    model.eval()

    if use_fp16:
        model.to(torch.float16)


    scheduler = lvdm_DDIM_Scheduler(model=model)

    pipeline = VC2_Pipeline_I2V_SpherePano(pretrained_t2v=model,
                                           scheduler=scheduler,
                                           model_config=model_config)
    pipeline.to(model.device)

    if use_fp16:
        pipeline.to(model.device, torch_dtype=torch.float16)

    # sample shape
    assert (run_args.height % 16 == 0) and (run_args.width % 16 == 0), "Error: image size [h,w] should be multiples of 16!"

    channels = model.channels

    batch_size = 1
    prompts = [prompt]
    img_cond_path = [pano_image_path]

    print("==== Sphere Panorama Shift Windows Sample ====")

    if vargs.predenoised_SP_latent_path is None:
        sphere_SW_latent, sphere_SW_denoised = pipeline.basic_sample_shift_shpere_panorama(
            prompt=prompts,
            img_cond_path=img_cond_path,
            height=run_args.height,
            width=run_args.width,
            frames=16, 
            fps=run_args.fps,
            guidance_scale=run_args.unconditional_guidance_scale,

            init_panorama_latent=None,
            use_skip_time=use_skip_time,
            skip_time_step_idx=skip_time_step_idx,
            progressive_skip=progressive_skip,

            loop_step=loop_step,
            pano_image_path=pano_image_path,

            total_f=total_f,
            dock_at_f=dock_at_f,
            overlap_ratio_list_f=overlap_ratio_list_1_f,
            loop_step_frame=loop_step_frame,

            equirect_width=equirect_width * upscale_factor if vargs.skip_1x else equirect_width * 2,    # 避免过大导致motion偏小?
            equirect_height=equirect_height * upscale_factor if vargs.skip_1x else equirect_height * 2,
            phi_theta_dict=phi_theta_dict,
            phi_prompt_dict=phi_prompt_dict,
            view_fov=view_fov,
            loop_step_theta=loop_step_theta,
            merge_renoised_overlap_latent_ratio=merge_renoised_overlap_latent_ratio,

            paste_on_static=paste_on_static,

            view_get_scale_factor=view_get_scale_factor,
            view_set_scale_factor=view_set_scale_factor,

            denoise_to_step=denoise_to_step,
            merge_prev_denoised_ratio_list=merge_prev_denoised_ratio_list,

            downsample_factor_before_vae_decode=downsample_factor_before_vae_decode,
            latents=None,
            num_inference_steps=run_args.num_inference_steps,
            num_videos_per_prompt=1,
            generator_seed=run_args.seed,

            output_type = "latent"
        )

        if save_latents:
            torch.save(sphere_SW_latent, os.path.join(output_dir, "sphere_SW_latent.pt"))
            # torch.save(basic_SW_video_frames, os.path.join(output_dir, "basic_SW_video_frames.pt"))
    else:
        print(f"loading SW latent from {vargs.predenoised_SP_latent_path}")
        sphere_SW_latent = torch.load(vargs.predenoised_SP_latent_path)

    print("==== Normal Plane Shift Windows Sample ====")

    if not vargs.skip_1x:

        if vargs.predenoised_SW_1x_latent_path is None:

            downsampled_sphere_SW_latent = resize_video_latent(input_latent=sphere_SW_latent.clone(), mode="nearest",
                                                               target_height=int(equirect_height // downsample_factor_before_vae_decode // 8),
                                                               target_width=int(equirect_width // downsample_factor_before_vae_decode // 8))

            basic_SW_video_frames, basic_SW_latent = pipeline.basic_sample_shift_multi_windows(
                prompt=prompts,
                img_cond_path=img_cond_path,
                height=run_args.height,
                width=run_args.width,
                frames=16, 
                fps=run_args.fps,
                guidance_scale=run_args.unconditional_guidance_scale,

                init_panorama_latent=downsampled_sphere_SW_latent,
                use_skip_time=True,
                skip_time_step_idx=denoise_to_step,
                progressive_skip=False,
                total_h=int(equirect_height // downsample_factor_before_vae_decode),
                total_w=int(equirect_width // downsample_factor_before_vae_decode),
                num_windows_h=num_windows_h_2,
                num_windows_w=num_windows_w_2,
                num_windows_f=num_windows_f,
                loop_step=loop_step,
                pano_image_path=pano_image_path,

                total_f=total_f,
                dock_at_f=dock_at_f,
                overlap_ratio_list_f=overlap_ratio_list_1_f,
                loop_step_frame=loop_step_frame,

                merge_prev_denoised_ratio_list=merge_prev_denoised_ratio_list,

                latents=None,
                num_inference_steps=run_args.num_inference_steps,
                num_videos_per_prompt=1,
                generator_seed=run_args.seed,
            )

            if save_latents:
                torch.save(basic_SW_latent, os.path.join(output_dir, f"basic_SW_latent-{project_name}.pt"))
                torch.save(basic_SW_video_frames, os.path.join(output_dir, f"basic_SW_video_frames-{project_name}.pt"))

            save_decoded_video_latents(decoded_video_latents=basic_SW_video_frames,
                                       output_path=output_dir,
                                       output_name="shift_windows",
                                       fps=run_args.fps)
        else:
            print(f"loading basic_SW_latent from : {vargs.predenoised_SW_1x_latent_path}")
            basic_SW_latent = torch.load(vargs.predenoised_SW_1x_latent_path)


    if vargs.do_upscale:
        print("==== Upscale Shift Windows Sample ====")

        if vargs.skip_1x:
            mixed_upscale_latent = sphere_SW_latent
        else:

            upsampled_SW_latent = resize_video_latent(input_latent=basic_SW_latent.clone(), mode="bicubic",
                                                      target_height=int(equirect_height // downsample_factor_before_vae_decode // 8 * upscale_factor),
                                                      target_width=int(equirect_width // downsample_factor_before_vae_decode // 8 * upscale_factor))
            pipeline.scheduler.make_schedule(run_args.num_inference_steps)
            renoised_basic_SW_latent = pipeline.scheduler.re_noise(x_a=upsampled_SW_latent,
                                                                   step_a=0,
                                                                   step_b=run_args.num_inference_steps-denoise_to_step)

            mixed_upscale_latent = renoised_basic_SW_latent 

        basic_SW_video_frames_2x, basic_SW_latent_2x = pipeline.basic_sample_shift_multi_windows(
            prompt=prompts,
            img_cond_path=img_cond_path,
            height=run_args.height,
            width=run_args.width,
            frames=16, 
            fps=run_args.fps,
            guidance_scale=run_args.unconditional_guidance_scale,

            init_panorama_latent=mixed_upscale_latent,
            use_skip_time=True,
            skip_time_step_idx=denoise_to_step,
            progressive_skip=False,
            total_h=int(equirect_height // downsample_factor_before_vae_decode * upscale_factor),
            total_w=int(equirect_width // downsample_factor_before_vae_decode * upscale_factor),
            num_windows_h=num_windows_h_2 * upscale_factor,
            num_windows_w=num_windows_w_2 * upscale_factor,
            num_windows_f=num_windows_f,
            loop_step=loop_step,
            pano_image_path=pano_image_path,

            merge_prev_denoised_ratio_list=merge_prev_denoised_ratio_list,

            total_f=total_f,
            dock_at_f=dock_at_f,
            overlap_ratio_list_f=overlap_ratio_list_2_f,
            loop_step_frame=loop_step_frame,

            latents=None,
            num_inference_steps=run_args.num_inference_steps,
            num_videos_per_prompt=1,
            generator_seed=run_args.seed,
        )

        if save_latents:
            torch.save(basic_SW_latent_2x, os.path.join(output_dir, f"denoised_latent2x-{project_name}.pt"))

        save_decoded_video_latents(decoded_video_latents=basic_SW_video_frames_2x,
                                   output_path=output_dir,
                                   output_name=f"SW_2X_{project_name}",
                                   fps=run_args.fps)



if __name__ == "__main__":

    run_args = RunArgs()
    run_args.seed = vargs.seed
    run_args.base_ckpt_path = "./videocrafter_models/i2v_512_v1/model.ckpt"
    run_args.num_inference_steps = 48
    run_args.fps = 8
    prompt = vargs.prompt

    image_folder_name = None

    image_path = None
    image_folder = None

    pano_image_path = vargs.pano_image_path

    loop_step = vargs.loop_step_hw
    num_windows_h = None
    num_windows_w = None
    num_windows_f = 1

    # for Sphere Pano Denoising Phrase
    skip_time_step_idx = vargs.skip_time_step
    if skip_time_step_idx >= 0:
        use_skip_time = True
        progressive_skip = True
    else:
        use_skip_time = False
        skip_time_step_idx = 0
        progressive_skip = False    

    denoise_to_step = vargs.denoise_to_step        


    # Sphere Pano Basic
    downsample_factor_before_vae_decode = 1
    equirect_width = int(1024 * downsample_factor_before_vae_decode)
    equirect_height = int(512 * downsample_factor_before_vae_decode)

    view_fov = vargs.view_fov #

    phi_0_first = False

    phi_num = vargs.phi_num

    phi_theta_dict = {
        90:  [0],
        -90: [0],

        75:     [360*t//phi_num for t in range(phi_num)],
        -75:    [360*t//phi_num for t in range(phi_num)],
        60:     [360*t//phi_num for t in range(phi_num)],
        -60:    [360*t//phi_num for t in range(phi_num)],
        45:     [360*t//phi_num for t in range(phi_num)],
        -45:    [360*t//phi_num for t in range(phi_num)],
        0:      [360*t//phi_num for t in range(phi_num)],
    }

    if phi_0_first:
        phi_theta_dict = OrderedDict(reversed(list(phi_theta_dict.items())))
    phi_prompt_dict = vargs.phi_prompt_dict


    paste_on_static = True
    loop_step_theta = vargs.loop_step_theta # Sphere Pano SW

    merge_renoised_overlap_latent_ratio = vargs.merge_renoised_overlap_latent_ratio

    view_get_scale_factor = 1
    view_set_scale_factor = 1

    num_windows_h_2 = 2
    num_windows_w_2 = 2

    total_f = vargs.total_f
    dock_at_f = vargs.dock_at_f
    loop_step_frame = vargs.loop_step_frame

    overlap_ratio_list_1_f_org = [0.75, 0.5]
    overlap_ratio_list_1_f = [
        overlap_ratio_list_1_f_org[i * len(overlap_ratio_list_1_f_org) // run_args.num_inference_steps] for i in
        range(run_args.num_inference_steps)]
    print(f"overlap_ratio_list for 1x F: {overlap_ratio_list_1_f}")

    overlap_ratio_list_2_f_org = [0.75, 0.5]
    overlap_ratio_list_2_f = [overlap_ratio_list_2_f_org[i * len(overlap_ratio_list_2_f_org) // run_args.num_inference_steps] for i in range(run_args.num_inference_steps)]
    print(f"overlap_ratio_list for 1x F: {overlap_ratio_list_2_f}")

    if vargs.merge_denoised:
        merge_prev_denoised_ratio_list = [vargs.max_merge_denoised_overlap_latent_ratio * (1 - t / vargs._merge_prev_step) for t in range(vargs._merge_prev_step)] + [0] * (run_args.num_inference_steps - vargs._merge_prev_step)

        print(f"merge_prev_denoised_ratio_list: {merge_prev_denoised_ratio_list}")
    else:
        merge_prev_denoised_ratio_list = None

    upscale_factor = vargs.upscale_factor

    PROJECT_FOLDER = "TEST_1"
    PROJECT_NOTE = f"s-{vargs.seed}"
    PROJECT_NAME = f"{PROJECT_NOTE}"

    save_latents = True

    main(run_args, prompt, image_path, image_folder,
         project_name=PROJECT_NAME,
         project_folder=PROJECT_FOLDER,

         pano_image_path=pano_image_path,
         loop_step=loop_step,
         num_windows_h=num_windows_h,
         num_windows_w=num_windows_w,
         num_windows_f=num_windows_f,

         use_skip_time=use_skip_time,
         skip_time_step_idx=skip_time_step_idx,
         progressive_skip=progressive_skip,

         equirect_width=equirect_width,
         equirect_height=equirect_height,
         phi_theta_dict=phi_theta_dict,
         phi_prompt_dict=phi_prompt_dict,

         view_fov=view_fov,
         loop_step_theta=loop_step_theta,
         merge_renoised_overlap_latent_ratio=merge_renoised_overlap_latent_ratio,

         paste_on_static=paste_on_static,

         view_get_scale_factor=view_get_scale_factor,
         view_set_scale_factor=view_set_scale_factor,

         downsample_factor_before_vae_decode=downsample_factor_before_vae_decode,

         denoise_to_step=denoise_to_step,

         num_windows_h_2=num_windows_h_2,
         num_windows_w_2=num_windows_w_2,

         dock_at_f=dock_at_f,
         overlap_ratio_list_1_f=overlap_ratio_list_1_f,
         overlap_ratio_list_2_f=overlap_ratio_list_2_f,

         loop_step_frame=loop_step_frame,

         total_f=total_f,
         upscale_factor=upscale_factor,

         merge_prev_denoised_ratio_list=merge_prev_denoised_ratio_list,

         save_latents=save_latents)
