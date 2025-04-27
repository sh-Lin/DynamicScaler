import numpy as np
from tqdm import tqdm
import torch

from lvdm.models.utils_diffusion import make_ddim_sampling_parameters, make_ddim_timesteps

class lvdm_DDIM_Scheduler(object):


    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule
        self.counter = 0


    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):

        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)

        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.betas = to_torch(self.model.betas)
        self.alphas_cumprod = to_torch(alphas_cumprod)
        self.alphas_cumprod_prev = to_torch(self.model.alphas_cumprod_prev)
        self.use_scale = self.model.use_scale

        if self.use_scale:
            self.scale_arr = to_torch(self.model.scale_arr)
            ddim_scale_arr = self.scale_arr.cpu()[self.ddim_timesteps]
            self.ddim_scale_arr = ddim_scale_arr
            ddim_scale_arr = np.asarray([self.scale_arr.cpu()[0]] + self.scale_arr.cpu()[self.ddim_timesteps[:-1]].tolist())
            self.ddim_scale_arr_prev = ddim_scale_arr

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = to_torch(np.sqrt(alphas_cumprod.cpu()))
        self.sqrt_one_minus_alphas_cumprod = to_torch(np.sqrt(1. - alphas_cumprod.cpu()))
        self.log_one_minus_alphas_cumprod = to_torch(np.log(1. - alphas_cumprod.cpu()))
        self.sqrt_recip_alphas_cumprod = to_torch(np.sqrt(1. / alphas_cumprod.cpu()))
        self.sqrt_recipm1_alphas_cumprod = to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.ddim_sigmas = ddim_sigmas
        self.ddim_alphas = ddim_alphas
        self.ddim_alphas_prev = ddim_alphas_prev
        self.ddim_sqrt_one_minus_alphas = np.sqrt(1. - ddim_alphas)
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.ddim_sigmas_for_original_num_steps = sigmas_for_original_sampling_steps


    @torch.no_grad()                                        
    def ddim_step(self, sample, noise_pred, indices):       
        b, _, f, *_, device = *sample.shape, sample.device

        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep

        size = (b, 1, 1, 1, 1)

        x_prevs = []
        pred_x0s = []

        for i, index in enumerate(indices):
            x = sample[:, :, [i]]
            e_t = noise_pred[:, :, [i]]
            a_t = torch.full(size, alphas[index], device=device)
            a_prev = torch.full(size, alphas_prev[index], device=device)
            sigma_t = torch.full(size, sigmas[index], device=device)
            sqrt_one_minus_at = torch.full(size, sqrt_one_minus_alphas[index], device=device)
            # current prediction for x_0
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
            # direction pointing to x_t
            dir_xt = (1. - a_prev - sigma_t ** 2).sqrt() * e_t

            noise = sigma_t * torch.randn(x.shape, device=device)

            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
            x_prevs.append(x_prev)
            pred_x0s.append(pred_x0)

        x_prev = torch.cat(x_prevs, dim=2)
        pred_x0 = torch.cat(pred_x0s, dim=2)

        return x_prev, pred_x0

    def re_noise(self, x_a, step_a, step_b):
        timestep_a = self.ddim_timesteps[step_a]
        timestep_b = self.ddim_timesteps[step_b]
        alpha_cumprod = self.alphas_cumprod.to(x_a.device)
        alpha_cumprod_a = alpha_cumprod[timestep_a]
        alpha_cumprod_b = alpha_cumprod[timestep_b]
        c = torch.sqrt(alpha_cumprod_b / alpha_cumprod_a)
        s = torch.sqrt(1 - alpha_cumprod_b / alpha_cumprod_a)
        epsilon_tilde = torch.randn_like(x_a)

        x_b = c * x_a + s * epsilon_tilde

        return x_b
