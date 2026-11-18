"""CPU prediction adapters for two publicly documented expression models.

Each output stores scores and input IDs/hashes only. The predictor does not
read either split's label array. Score order is FER2013's native seven classes.
"""
from __future__ import annotations

from pathlib import Path
import sys

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / 'public_resources' / 'prepared'
CLASS_ORDER = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')


class AffectLens:
    """FER2013 ResNet18 trained with FER+ blended votes and grayscale inputs."""
    filename = 'affectlens.pt'

    def __init__(self, checkpoint: Path, threads: int):
        import torch
        sys.path.insert(0, str(HERE / 'affectlens_source' / 'src'))
        from affectlens.model import AffectLensNet

        self.torch = torch
        torch.set_num_threads(threads)
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        config = dict(state['model_config'])
        config['pretrained'] = False
        self.model = AffectLensNet(**config).eval()
        self.model.load_state_dict(state['model_state'], strict=True)
        self.model = self.model.to(memory_format=torch.channels_last)
        train_config = state['train_config']
        self.size = train_config['image_size']
        self.normalization = train_config.get('normalization', 'symmetric')
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)) if train_config.get('contrast_normalization') else None

    def forward_outputs(self, images: np.ndarray):
        torch = self.torch
        frames = [self.clahe.apply(gray) for gray in images] if self.clahe is not None else images
        frames = np.stack([cv2.resize(gray, (self.size, self.size), interpolation=cv2.INTER_CUBIC) for gray in frames])
        with torch.inference_mode():
            x = torch.from_numpy(frames).unsqueeze(1).float().div_(255)
            mean, std = (0.449, 0.226) if self.normalization == 'imagenet_gray' else (0.5, 0.5)
            x.sub_(mean).div_(std)
            return self.model(x.contiguous(memory_format=torch.channels_last))

    def __call__(self, images: np.ndarray) -> np.ndarray:
        return self.forward_outputs(images).logits.numpy()


class OxfordFER:
    """Oxford Learning Grimaces models trained on original FER2013 labels."""

    def __init__(self, checkpoint: Path, threads: int):
        import torch
        from torch.nn.utils.fusion import fuse_conv_bn_eval

        self.torch = torch
        torch.set_num_threads(threads)
        self.model = self.create_model().eval()
        weights = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.model.load_state_dict(weights, strict=True)
        del weights
        convolutions = ('conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'fc6', 'fc7')
        for index, convolution in enumerate(convolutions, start=self.first_norm_index):
            norm = f'bn{index}'
            setattr(self.model, convolution, fuse_conv_bn_eval(getattr(self.model, convolution), getattr(self.model, norm)))
            setattr(self.model, norm, torch.nn.Identity())
        self.model = self.model.to(memory_format=torch.channels_last)
        self.size = tuple(self.model.meta['imageSize'][:2])
        self.mean = torch.tensor(self.model.meta['mean'])[None, :, None, None]

    def __call__(self, images: np.ndarray) -> np.ndarray:
        from PIL import Image
        torch = self.torch
        gray = np.stack([np.asarray(Image.fromarray(im).resize(self.size, Image.Resampling.BILINEAR)) for im in images])
        with torch.inference_mode():
            rgb = torch.from_numpy(gray).unsqueeze(1).float().expand(-1, 3, -1, -1)
            return self.model((rgb-self.mean).contiguous(memory_format=torch.channels_last)).numpy()


class OxfordAlexNet(OxfordFER):
    filename = 'oxford_alexnet.pth'
    first_norm_index = 1

    @staticmethod
    def create_model():
        from alexnet_face_fer_bn_dag import Alexnet_face_fer_bn_dag

        return Alexnet_face_fer_bn_dag()


class OxfordVGGM(OxfordFER):
    filename = 'oxford_vggm.pth'
    first_norm_index = 49

    @staticmethod
    def create_model():
        from vgg_m_face_bn_fer_dag import Vgg_m_face_bn_fer_dag

        return Vgg_m_face_bn_fer_dag()


