"""CPU preprocessing and inference for the selected FER2013 checkpoints."""
import contextlib

import cv2
import numpy as np
import torch
from torch import nn

def fuse(model):
    from torch.nn.utils.fusion import fuse_conv_bn_eval
    for child in model.children():
        fuse(child)
    for conv, bn in [("conv", "bn"), ("conv1", "bn1"), ("conv2", "bn2"), ("conv3", "bn3")]:
        convolution = getattr(model, conv, None)
        batchnorm = getattr(model, bn, None)
        if isinstance(convolution, nn.Conv2d) and isinstance(batchnorm, nn.BatchNorm2d):
            setattr(model, conv, fuse_conv_bn_eval(convolution, batchnorm))
            setattr(model, bn, nn.Identity())
    if isinstance(model, nn.Sequential):
        for i in range(len(model) - 1):
            if isinstance(model[i], nn.Conv2d) and isinstance(model[i + 1], nn.BatchNorm2d):
                model[i] = fuse_conv_bn_eval(model[i], model[i + 1])
                model[i + 1] = nn.Identity()

def image_transform(image, size, angle, flip):
    pixels = cv2.resize(image, (size, size), interpolation=cv2.INTER_LINEAR)
    if flip:
        pixels = pixels[:, ::-1]
    if angle:
        matrix = cv2.getRotationMatrix2D(((size - 1) / 2, (size - 1) / 2), angle, 1)
        pixels = cv2.warpAffine(pixels, matrix, (size, size), flags=cv2.INTER_LINEAR)
    return np.repeat(pixels[..., None], 3, -1)

def run(model, images, size, views, dtype, batch_size, name):
    blocks = []
    for begin in range(0, len(images), batch_size):
        batch = images[begin : begin + batch_size]
        all_views = []
        for angle, flip in views:
            if name == "tyom_resnet18":
                zoom = angle
                batch_tensor = torch.from_numpy(batch[:, None].copy()).float().div_(255)
                theta = batch_tensor.new_tensor([[-1 / zoom if flip else 1 / zoom, 0, 0], [0, 1 / zoom, 0]])
                grid = torch.nn.functional.affine_grid(theta.expand(len(batch), -1, -1), (len(batch), 1, size, size), align_corners=False)
                tensor = torch.nn.functional.grid_sample(batch_tensor, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
                mean = tensor.new_tensor([.485, .456, .406])[None, :, None, None]
                std = tensor.new_tensor([.229, .224, .225])[None, :, None, None]
                tensor = ((tensor - mean) / std).contiguous(memory_format=torch.channels_last)
            else:
                pixels = np.stack([image_transform(im, size, angle, flip) for im in batch])
                tensor = torch.from_numpy(pixels).permute(0, 3, 1, 2).float().div_(255)
            if dtype == "int8":
                predictions = torch.from_numpy(model.run(None, {"image": tensor.numpy()})[0])
            else:
                with contextlib.nullcontext() if dtype == "fp32" else torch.autocast("cpu", dtype=torch.bfloat16):
                    predictions = model(tensor)
            if name == "tyom_resnet18":
                predictions = predictions[:, [0, 1, 2, 3, 5, 6, 4]]
            all_views.append(predictions.to(dtype=torch.float32).numpy())
        blocks.append(np.stack(all_views, axis=1))
    return np.concatenate(blocks)

