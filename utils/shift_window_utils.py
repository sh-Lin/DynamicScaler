
import torch
import numpy as np
import cv2
from PIL import Image

from lvdm.models.ddpm3d import LatentVisualDiffusion

import torch

from utils.tensor_utils import load_image_tensor_from_path


def get_dimension_slices_and_sizes(begin, end, size):
    total_length = end - begin
    slices = []
    sizes = []
    current_pos = begin

    while current_pos < end:
        start_idx = current_pos % size
        next_boundary = ((current_pos // size) + 1) * size
        end_pos = min(end, next_boundary)
        length = end_pos - current_pos
        end_idx = (start_idx + length) % size

        if end_idx > start_idx:
            slices.append(slice(start_idx, end_idx))
            sizes.append(end_idx - start_idx)
        else:
            slices.append(slice(start_idx, size))
            sizes.append(size - start_idx)
            if end_idx > 0:
                slices.append(slice(0, end_idx))
                sizes.append(end_idx)
        current_pos = end_pos

    return slices, sizes

class RingLatent:
    def __init__(self, init_latent):
        self.torch_latent = init_latent.clone()
        assert len(self.torch_latent.shape) == 5, f"[RingLatent.__init__] init_latent shape {init_latent.shape} not legal"

    def get_shape(self):
        return self.torch_latent.shape

    def get_window_latent(self,
                          pos_left: int = None,
                          pos_right: int = None,
                          pos_top: int = None,
                          pos_down: int = None,
                          frame_begin: int = None,
                          frame_end: int = None):
        if pos_left is None:
            pos_left = 0
        if pos_right is None:
            pos_right = self.get_shape()[-1]
        if pos_top is None:
            pos_top = 0
        if pos_down is None:
            pos_down = self.get_shape()[-2]
        if frame_begin is None:
            frame_begin = 0
        if frame_end is None:
            frame_end = self.get_shape()[2]  # frame 维度的索引是2

        depth = self.get_shape()[2]   # frame 维度的大小
        height = self.get_shape()[-2]
        width = self.get_shape()[-1]

        # 确保索引位置是合法的
        assert 0 <= pos_left < pos_right <= width * 2, f"Invalid pos_left {pos_left} and pos_right {pos_right}"
        assert 0 <= pos_top < pos_down <= height * 2, f"Invalid pos_top {pos_top} and pos_down {pos_down}"
        assert 0 <= frame_begin < frame_end <= depth * 2, f"Invalid frame_begin {frame_begin} and frame_end {frame_end}"

        # # 处理宽度维度的索引
        # if pos_right <= width:
        #     width_slices = [slice(pos_left, pos_right)]
        # else:
        #     width_slices = [slice(pos_left, width), slice(0, pos_right % width)]
        #
        # # 处理高度维度的索引
        # if pos_down <= height:
        #     height_slices = [slice(pos_top, pos_down)]
        # else:
        #     height_slices = [slice(pos_top, height), slice(0, pos_down % height)]
        #
        # # 处理 frame 维度的索引
        # if frame_end <= depth:
        #     frame_slices = [slice(frame_begin, frame_end)]
        # else:
        #     frame_slices = [slice(frame_begin, depth), slice(0, frame_end % depth)]

        width_slices, width_sizes = get_dimension_slices_and_sizes(pos_left, pos_right, width)
        height_slices, height_sizes = get_dimension_slices_and_sizes(pos_top, pos_down, height)
        frame_slices, frame_sizes = get_dimension_slices_and_sizes(frame_begin, frame_end, depth)

        # 收集各部分数据
        parts = []
        for f_slice in frame_slices:
            frame_parts = []
            for h_slice in height_slices:
                row_parts = []
                for w_slice in width_slices:
                    part = self.torch_latent[:, :, f_slice, h_slice, w_slice]
                    row_parts.append(part)
                row = torch.cat(row_parts, dim=4)  # 在宽度维度上拼接
                frame_parts.append(row)
            frame_block = torch.cat(frame_parts, dim=3)  # 在高度维度上拼接
            parts.append(frame_block)
        desired_latent = torch.cat(parts, dim=2)  # 在 frame 维度上拼接

        return desired_latent.clone()

    def set_window_latent(self, input_latent: torch.Tensor,
                          pos_left: int = None,
                          pos_right: int = None,
                          pos_top: int = None,
                          pos_down: int = None,
                          frame_begin: int = None,
                          frame_end: int = None):
        if pos_left is None:
            pos_left = 0
        if pos_right is None:
            pos_right = self.get_shape()[-1]
        if pos_top is None:
            pos_top = 0
        if pos_down is None:
            pos_down = self.get_shape()[-2]
        if frame_begin is None:
            frame_begin = 0
        if frame_end is None:
            frame_end = self.get_shape()[2]

        depth = self.get_shape()[2]
        height = self.get_shape()[-2]
        width = self.get_shape()[-1]

        # 确保索引位置是合法的
        assert 0 <= pos_left < pos_right <= width * 2, f"Invalid pos_left {pos_left} and pos_right {pos_right}"
        assert 0 <= pos_top < pos_down <= height * 2, f"Invalid pos_top {pos_top} and pos_down {pos_down}"
        assert 0 <= frame_begin < frame_end <= depth * 2, f"Invalid frame_begin {frame_begin} and frame_end {frame_end}"

        assert pos_right - pos_left <= width, f"warp should not occur"
        assert pos_down - pos_top <= height, f"warp should not occur"
        assert frame_end - frame_begin <= depth, f"warp should not occur"

        # 计算目标窗口的深度、高度和宽度
        target_depth = frame_end - frame_begin if frame_end <= depth else (depth - frame_begin) + (frame_end % depth)
        target_height = pos_down - pos_top if pos_down <= height else (height - pos_top) + (pos_down % height)
        target_width = pos_right - pos_left if pos_right <= width else (width - pos_left) + (pos_right % width)

        # # 处理宽度维度的索引和尺寸
        # if pos_right <= width:
        #     width_slices = [slice(pos_left, pos_right)]
        #     width_sizes = [pos_right - pos_left]
        # else:
        #     width_slices = [slice(pos_left, width), slice(0, pos_right % width)]
        #     width_sizes = [width - pos_left, pos_right % width]
        #
        # # 处理高度维度的索引和尺寸
        # if pos_down <= height:
        #     height_slices = [slice(pos_top, pos_down)]
        #     height_sizes = [pos_down - pos_top]
        # else:
        #     height_slices = [slice(pos_top, height), slice(0, pos_down % height)]
        #     height_sizes = [height - pos_top, pos_down % height]
        #
        # # 处理 frame 维度的索引和尺寸
        # if frame_end <= depth:
        #     frame_slices = [slice(frame_begin, frame_end)]
        #     frame_sizes = [frame_end - frame_begin]
        # else:
        #     frame_slices = [slice(frame_begin, depth), slice(0, frame_end % depth)]
        #     frame_sizes = [depth - frame_begin, frame_end % depth]


        width_slices, width_sizes = get_dimension_slices_and_sizes(pos_left, pos_right, width)
        height_slices, height_sizes = get_dimension_slices_and_sizes(pos_top, pos_down, height)
        frame_slices, frame_sizes = get_dimension_slices_and_sizes(frame_begin, frame_end, depth)

        # Calculate the target window shape
        target_depth = sum(frame_sizes)
        target_height = sum(height_sizes)
        target_width = sum(width_sizes)


        # 检查输入张量的形状是否匹配
        assert input_latent.shape[2:] == (target_depth, target_height, target_width), f"Input latent shape {input_latent.shape[2:]} does not match target window shape {(target_depth, target_height, target_width)}"


        # 将输入张量按 frame、高度和宽度分割，并写入原始张量
        f_start = 0
        for f_slice, f_size in zip(frame_slices, frame_sizes):
            h_start = 0
            for h_slice, h_size in zip(height_slices, height_sizes):
                w_start = 0
                for w_slice, w_size in zip(width_slices, width_sizes):
                    # 从输入张量中取出对应部分
                    input_part = input_latent[:, :, f_start:f_start+f_size, h_start:h_start+h_size, w_start:w_start+w_size]
                    # 写入原始张量的对应位置
                    self.torch_latent[:, :, f_slice, h_slice, w_slice] = input_part
                    w_start += w_size
                h_start += h_size
            f_start += f_size


class RingImageTensor:

    def __init__(self,
                 image_path: str,
                 image_tensor: torch.Tensor = None,
                 height: int = 320,
                 width: int = 512,
                 ):

        if image_tensor is None:
            self.image_tensor = load_image_tensor_from_path(image_path, height, width)
        else:
            self.image_tensor = image_tensor

        assert list(self.image_tensor.shape) == [3, height, width], f"[RingImageTensor] image shape {self.image_tensor.shape} " \
                                                                    f"does not match {[3, height, width]}"

    def get_shape(self):
        """
        shape: [3, h, w]
        """
        return self.image_tensor.shape

    def get_window_tensor(self,
                          pos_left: int,
                          pos_right: int,
                          pos_top: int = None,
                          pos_down: int = None):

        if pos_top is None:
            pos_top = 0
        if pos_down is None:
            pos_down = self.get_shape()[-2]

        height = self.get_shape()[-2]
        width = self.get_shape()[-1]

        # Ensure indices are within legal bounds
        assert 0 <= pos_left < pos_right <= width * 2, f"[RingImageTensor.get_window_tensor] pos_left {pos_left}, pos_right {pos_right} not legal"
        assert 0 <= pos_top < pos_down <= height * 2, f"[RingImageTensor.get_window_tensor] pos_top {pos_top}, pos_down {pos_down} not legal"

        # Calculate slices
        width_slices, width_sizes = get_dimension_slices_and_sizes(pos_left, pos_right, width)
        height_slices, height_sizes = get_dimension_slices_and_sizes(pos_top, pos_down, height)

        # Collect parts and concatenate
        parts = []
        for h_slice in height_slices:
            row_parts = []
            for w_slice in width_slices:
                part = self.image_tensor[:, h_slice, w_slice]
                row_parts.append(part)
            row = torch.cat(row_parts, dim=2)  # Concatenate along width dimension
            parts.append(row)
        desired_tensor = torch.cat(parts, dim=1)  # Concatenate along height dimension

        return desired_tensor

    def get_encoded_image_cond(self,
                               pretrained_t2v: LatentVisualDiffusion,
                               pos_left: int,
                               pos_right: int,
                               pos_top: int = None,
                               pos_down: int = None):
        cond_image_tensor = self.get_window_tensor(pos_left=pos_left, pos_right=pos_right, pos_top=pos_top, pos_down=pos_down)
        cond_image_tensor = cond_image_tensor.to(pretrained_t2v.device).unsqueeze(dim=0)  # `get_image_embeds` expects [b, c, h, w]
        img_emb = pretrained_t2v.get_image_embeds(cond_image_tensor)
        return img_emb

    # def _load_image_tensor_from_path(self,
    #                                  image_path: str,
    #                                  height: int,
    #                                  width: int,
    #                                  norm_to_1 = True):
    #     img = Image.open(image_path).convert("RGB")
    #     rgb_img = np.array(img, np.float32)
    #     rgb_img = cv2.resize(rgb_img, (width, height), interpolation=cv2.INTER_LINEAR)
    #     img_tensor = torch.from_numpy(rgb_img).permute(2, 0, 1)  # .float()
    #
    #     if norm_to_1:
    #         img_tensor = (img_tensor / 255. - 0.5) * 2  # Normalize to [-1, 1]
    #
    #     return img_tensor



