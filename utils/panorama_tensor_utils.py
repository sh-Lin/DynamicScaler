
import torch
import torch.nn.functional as F

class PanoramaTensor:
    def __init__(self, equirect_tensor):
        assert equirect_tensor.dim() >= 2
        H, W = equirect_tensor.shape[-2], equirect_tensor.shape[-1]
        assert W == 2 * H

        if equirect_tensor.dim() == 2:
            equirect_tensor = equirect_tensor.unsqueeze(0)  # [1, H, W]

        C = equirect_tensor.shape[-3] if equirect_tensor.dim() >= 3 else 1
        if equirect_tensor.dim() == 3:
            C = equirect_tensor.shape[0]
        elif equirect_tensor.dim() > 3:
            C = equirect_tensor.shape[-3]

        self.equirect_tensor = equirect_tensor.clone()
        self.C = C
        self.H = H
        self.W = W
        self.device = equirect_tensor.device
        self.dtype = equirect_tensor.dtype


    def get_view_tensor_interpolate(self, fov, theta, phi, width, height,
                                    interpolate_mode='bilinear', interpolate_align_corners=True):
        leading_dims = self.equirect_tensor.shape[:-3] if self.equirect_tensor.dim() > 3 else ()
        B = torch.prod(torch.tensor(leading_dims)) if len(leading_dims) > 0 else 1
        B = int(B)

        pano = self.equirect_tensor.view(-1, self.C, self.H, self.W)  # [B, C, H, W]

        u, v = self._get_uv(fov=fov, theta=theta, phi=phi, width=width, height=height)

        grid_u = (u / (self.W - 1)) * 2 - 1  # [height, width]
        grid_v = (v / (self.H - 1)) * 2 - 1  # [height, width]
        grid = torch.stack((grid_u, grid_v), dim=-1)  # [height, width, 2]
        grid = grid.unsqueeze(0).repeat(B, 1, 1, 1)  # [B, height, width, 2]

        view = F.grid_sample(pano, grid, mode=interpolate_mode, padding_mode='border',
                             align_corners=interpolate_align_corners)  # [B, C, height, width]

        if len(leading_dims) > 0:
            view = view.view(*leading_dims, self.C, height, width)  # [*, C, height, width]
        else:
            view = view.squeeze(0)  # [C, height, width]

        return view

    def get_view_tensor_no_interpolate(self, fov, theta, phi, width, height):

        leading_dims = self.equirect_tensor.shape[:-3] if self.equirect_tensor.dim() > 3 else ()
        B = torch.prod(torch.tensor(leading_dims)) if len(leading_dims) > 0 else 1
        B = int(B)

        pano = self.equirect_tensor.view(-1, self.C, self.H, self.W)  # [B, C, H, W]

        u, v = self._get_uv(fov=fov, theta=theta, phi=phi, width=width, height=height)

        sampled_view, unsampled_mask = self._sample_equirect_tensor_nearest(pano, u, v)

        if len(leading_dims) > 0:
            sampled_view = sampled_view.view(*leading_dims, self.C, height, width)  # [*, C, height, width]
        else:
            sampled_view = sampled_view.squeeze(0)  # [C, height, width]

        return sampled_view, unsampled_mask

    def set_view_tensor(self, view_tensor, fov, theta, phi):

        if view_tensor.dim() == 3:
            view_tensor = view_tensor.unsqueeze(0)  # [1, C, H, W]

        leading_dims = self.equirect_tensor.shape[:-3]  # [*, C, H, W]
        B = torch.prod(torch.tensor(leading_dims)) if len(leading_dims) > 0 else 1
        B = int(B)

        pano = self.equirect_tensor.view(-1, self.C, self.H, self.W)  # [B, C, H, W]
        view = view_tensor.view(-1, self.C, view_tensor.shape[-2], view_tensor.shape[-1]).clone()  # [B, C, height, width]

        width, height = view.shape[-1], view.shape[-2]

        u, v = self._get_uv(fov=fov, theta=theta, phi=phi, width=width, height=height)
        u_nn = torch.round(u).long().clamp(0, self.W - 1)
        v_nn = torch.round(v).long().clamp(0, self.H - 1)
        flat_view = view.view(B, self.C, -1)  # [B, C, height*width]
        flat_pano = pano.view(B, self.C, -1)  # [B, C, H_pano*W_pano]

        linear_indices = (v_nn * self.W + u_nn).view(B, -1)  # [B, height*width]
        flat_pano.scatter_(2, linear_indices.unsqueeze(1).expand(-1, self.C, -1), flat_view)

        pano = flat_pano.view(B, self.C, self.H, self.W)
        self.equirect_tensor = pano.view(*leading_dims, self.C, self.H, self.W) if B > 1 else pano.squeeze(0)

    def set_view_tensor_bilinear(self, view_tensor, fov, theta, phi):
        if view_tensor.dim() == 3:
            view_tensor = view_tensor.unsqueeze(0)  # [1, C, H, W]

        leading_dims = view_tensor.shape[:-3]  # [*, C, H, W]
        B = torch.prod(torch.tensor(leading_dims)).item() if len(leading_dims) > 0 else 1
        view = view_tensor.view(B, self.C, view_tensor.shape[-2], view_tensor.shape[-1]).clone()  # [B, C, height, width]

        width, height = view.shape[-1], view.shape[-2]

        u, v = self._get_uv(fov=fov, theta=theta, phi=phi, width=width, height=height)

        u0 = torch.floor(u).long()
        v0 = torch.floor(v).long()
        u1 = (u0 + 1) % self.W
        v1 = torch.clamp(v0 + 1, 0, self.H - 1)

        du = (u - u0.float()).unsqueeze(0)
        dv = (v - v0.float()).unsqueeze(0)

        w00 = (1 - du) * (1 - dv)
        w01 = (1 - du) * dv
        w10 = du * (1 - dv)
        w11 = du * dv

        view_flat = view.view(B, self.C, -1)
        w00 = w00.view(-1)  # [height*width]
        w01 = w01.view(-1)
        w10 = w10.view(-1)
        w11 = w11.view(-1)

        idx00 = (v0 * self.W + u0).view(-1)  # [height*width]
        idx01 = (v1 * self.W + u0).view(-1)
        idx10 = (v0 * self.W + u1).view(-1)
        idx11 = (v1 * self.W + u1).view(-1)

        for b in range(B):

            accumulator = torch.zeros_like(self.equirect_tensor.view(B, self.C, -1)[b])  # [B, C, H_pano*W_pano]
            weight_sum = torch.zeros_like(self.equirect_tensor.view(B, self.C, -1)[b])  # [B, C, H_pano*W_pano]

            for c in range(self.C):

                accumulator[c].index_add_(0, idx00, view_flat[b, c] * w00)
                accumulator[c].index_add_(0, idx01, view_flat[b, c] * w01)
                accumulator[c].index_add_(0, idx10, view_flat[b, c] * w10)
                accumulator[c].index_add_(0, idx11, view_flat[b, c] * w11)

                weight_sum[c].index_add_(0, idx00, w00)
                weight_sum[c].index_add_(0, idx01, w01)
                weight_sum[c].index_add_(0, idx10, w10)
                weight_sum[c].index_add_(0, idx11, w11)

            mask = weight_sum > 0
            self.equirect_tensor.view(B, self.C, -1)[b][mask] = accumulator[mask] / weight_sum[mask]

    def set_view_tensor_no_interpolation(self, view_tensor, fov, theta, phi):
        if view_tensor.dim() == 3:
            view_tensor = view_tensor.unsqueeze(0)  # [1, C, H, W]

        leading_dims = view_tensor.shape[:-3]  # [*, C, H, W]
        B = torch.prod(torch.tensor(leading_dims)).item() if len(leading_dims) > 0 else 1

        view = view_tensor.view(B, self.C, view_tensor.shape[-2], view_tensor.shape[-1])  # [B, C, height, width]

        width, height = view.shape[-1], view.shape[-2]

        u, v = self._get_uv(fov=fov, theta=theta, phi=phi, width=width, height=height)  # [height, width]

        u_int = torch.floor(u).long()
        v_int = torch.floor(v).long()

        valid_mask = (u_int >= 0) & (u_int < self.W) & (v_int >= 0) & (v_int < self.H)  # [height, width]

        view_flat = view.view(B, self.C, -1)  # [B, C, height*width]
        pano_flat = self.equirect_tensor.view(-1, self.C, self.H * self.W)  # [B, C, H*W]

        linear_indices = (v_int * self.W + u_int).view(-1)  # [B * height * width]

        valid_linear_indices = linear_indices[valid_mask.view(-1)]  # [num_valid_pixels]
        valid_view = view_flat.reshape(B * self.C, -1)[:, valid_mask.view(-1)]  # [B*C, num_valid_pixels]

        pano_flat = pano_flat.reshape(B * self.C, -1)  # [B*C, H*W]
        pano_flat[:, valid_linear_indices] = valid_view

        self.equirect_tensor = pano_flat.reshape(self.equirect_tensor.shape)

    def _sample_equirect_tensor_nearest(self, pano, u, v):

        u0 = torch.floor(u).long()
        v0 = torch.floor(v).long()

        u0 = u0 % self.W
        v0 = torch.clamp(v0, 0, self.H - 1)

        sampled_view = pano[:, :, v0, u0].clone()  # [B, C, height, width]

        unsampled_mask = torch.ones_like(u0, dtype=self.dtype, device=self.device)  # [height, width]

        valid_mask = (u >= 0) & (u < self.W) & (v >= 0) & (v < self.H)
        unsampled_mask[~valid_mask] = 0

        sampled_view[:, :, ~valid_mask] = 0

        return sampled_view, unsampled_mask

    def _get_uv(self, fov, theta, phi, width, height, focal_length=None):
        fov_rad = torch.deg2rad(torch.tensor(fov, dtype=self.dtype, device=self.device))
        theta_rad = torch.deg2rad(torch.tensor(theta, dtype=self.dtype, device=self.device))
        phi_rad = torch.deg2rad(torch.tensor(phi, dtype=self.dtype, device=self.device))

        if focal_length is None:
            f = 0.5 * width / torch.tan(fov_rad / 2)
        else:
            f = focal_length

        # width, height = view.shape[-1], view.shape[-2]
        x = torch.linspace(-width / 2, width / 2 - 1, steps=width, dtype=self.dtype, device=self.device)
        y = torch.linspace(-height / 2, height / 2 - 1, steps=height, dtype=self.dtype, device=self.device)
        yv, xv = torch.meshgrid(y, x, indexing='ij')  # [height, width]

        zv = torch.full_like(xv, f)
        xyz = torch.stack([xv, yv, zv], dim=-1)
        norm = torch.norm(xyz, dim=-1, keepdim=True)
        xyz_norm = xyz / norm 

        R_phi = torch.tensor([
            [1, 0, 0],
            [0, torch.cos(phi_rad), -torch.sin(phi_rad)],
            [0, torch.sin(phi_rad), torch.cos(phi_rad)]
        ], dtype=self.dtype, device=self.device)

        R_theta = torch.tensor([
            [torch.cos(theta_rad), 0, torch.sin(theta_rad)],
            [0, 1, 0],
            [-torch.sin(theta_rad), 0, torch.cos(theta_rad)]
        ], dtype=self.dtype, device=self.device)

        R = torch.matmul(R_theta, R_phi)  # [3, 3]

        xyz_rot = torch.matmul(xyz_norm.view(-1, 3), R.t()).view(height, width, 3)  # [height, width, 3]
        lon = torch.atan2(xyz_rot[..., 0], xyz_rot[..., 2])  # [-pi, pi]
        lat = torch.asin(xyz_rot[..., 1])  # [-pi/2, pi/2]
        lon = (lon + 2 * torch.pi) % (2 * torch.pi)  # [0, 2*pi)
        u = lon / (2 * torch.pi) * (self.W - 1)  # [height, width]
        v = (lat + torch.pi / 2) / torch.pi * (self.H - 1)  # [height, width]

        return u, v



class PanoramaLatentProxy:
    def __init__(self, equirect_tensor):

        assert equirect_tensor.dim() >= 4, "输入张量必须至少具有四个维度 [B, C, N, H, W]"
        self.original_shape = equirect_tensor.shape
        B, C, N, H, W = self.original_shape

        equirect_tensor_reordered = equirect_tensor.permute(0, 2, 1, 3, 4)

        self.panorama_tensor = PanoramaTensor(equirect_tensor_reordered)

    def get_view_tensor_interpolate(self, fov, theta, phi, width, height,
                                    interpolate_mode='bilinear', interpolate_align_corners=True):
        view = self.panorama_tensor.get_view_tensor_interpolate(
            fov, theta, phi, width, height, interpolate_mode, interpolate_align_corners)

        B, N, C, H, W = view.shape
        return view.permute(0, 2, 1, 3, 4).clone()

    def get_view_tensor_no_interpolate(self, fov, theta, phi, width, height):
        view, mask = self.panorama_tensor.get_view_tensor_no_interpolate(fov, theta, phi, width, height)

        B, N, C, H, W = view.shape
        view = view.permute(0, 2, 1, 3, 4)

        return view, mask

    def set_view_tensor(self, view_tensor, fov, theta, phi):
        view_tensor_reordered = view_tensor.permute(0, 2, 1, 3, 4)
        self.panorama_tensor.set_view_tensor(view_tensor_reordered, fov, theta, phi)

    def set_view_tensor_bilinear(self, view_tensor, fov, theta, phi):
        view_tensor_reordered = view_tensor.permute(0, 2, 1, 3, 4)
        self.panorama_tensor.set_view_tensor_bilinear(view_tensor_reordered, fov, theta, phi)

    def get_equirect_tensor(self):
        equirect_tensor = self.panorama_tensor.equirect_tensor
        return equirect_tensor.permute(0, 2, 1, 3, 4)

    def set_view_tensor_no_interpolation(self, view_tensor, fov, theta, phi):
        view_tensor_reordered = view_tensor.permute(0, 2, 1, 3, 4)
        self.panorama_tensor.set_view_tensor_no_interpolation(view_tensor_reordered, fov, theta, phi)

