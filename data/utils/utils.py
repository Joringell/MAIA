import random

import torch


def intrinsic_transform(intrinsic, original_res, size, transform_mode):
    fx, fy = intrinsic[0, 0], intrinsic[1, 1]
    cx, cy = intrinsic[0, 2], intrinsic[1, 2]
    original_height = original_res[0]
    original_width = original_res[1]
    if transform_mode == 'resize':
        resize_height = size[0]
        resize_width = size[1]

        scale_height = resize_height / original_height
        scale_width = resize_width / original_width

        fx_new = fx * scale_width
        fy_new = fy * scale_height
        cx_new = cx * scale_width
        cy_new = cy * scale_height

    elif transform_mode == 'center_crop_resize':
        if original_height <= original_width:
            scale_ratio = min(size) / original_height
        else:
            scale_ratio = min(size) / original_width
        resize_height = scale_ratio * original_height
        resize_width = scale_ratio * original_width

        fx_new = fx * scale_ratio
        fy_new = fy * scale_ratio
        cx_new = cx * scale_ratio
        cy_new = cy * scale_ratio

        crop_height = size[0]
        crop_width = size[1]
        cx_new = cx_new * (crop_width / resize_width)
        cy_new = cy_new * (crop_height / resize_height)
    
    else:
        raise NotImplementedError('No such transformation mode for image!')
    
    return torch.tensor(
        [[fx_new, 0, cx_new], [0, fy_new, cy_new], [0, 0, 1]]
    )


def gen_crop_config(tensor):
    h, w = tensor.shape[-2], tensor.shape[-1]
    h_start = random.randint(0, h // 8)
    w_start = random.randint(0, w // 8)
    h_crop = random.randint(7 * h // 8, h - h_start)
    w_crop = random.randint(7 * w // 8, w - w_start)
    return h_start, w_start, h_crop, w_crop


def intrin_crop_transform(intrinsic, h_start, w_start):
    fx, fy = intrinsic[0][0], intrinsic[1][1]
    cx, cy = intrinsic[0][2], intrinsic[1][2]
    cx_new = cx - w_start
    cy_new = cy - h_start
    return torch.tensor([[fx, 0, cx_new], [0, fy, cy_new], [0, 0, 1]])
