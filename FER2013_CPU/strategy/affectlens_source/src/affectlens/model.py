"""Compact multi-task network for facial expression and affect estimation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


@dataclass
class ModelOutput:
    """Structured outputs from :class:`AffectLensNet`."""

    logits: Tensor
    valence_arousal: Tensor
    embedding: Tensor


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.net = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return x * self.net(self.pool(x))


class DepthwiseResidualBlock(nn.Module):
    """Mobile-friendly residual block with channel attention."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.expand = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                3,
                stride=stride,
                padding=1,
                groups=out_channels,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
            SqueezeExcitation(out_channels),
        )
        self.project = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.skip = (
            nn.Identity()
            if stride == 1 and in_channels == out_channels
            else nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = self.skip(x)
        x = self.project(self.depthwise(self.expand(x)))
        return F.silu(x + residual, inplace=True)


class AffectLensNet(nn.Module):
    """Predict seven expressions and continuous valence/arousal jointly.

    The network deliberately stays small enough for real-time webcam inference.
    Its final convolution is exposed as ``feature_layer`` for Grad-CAM.
    """

    def __init__(
        self,
        num_classes: int = 7,
        width: int = 32,
        embedding_dim: int = 128,
        dropout: float = 0.25,
        variant: str = "compact",
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        if variant not in {"compact", "resnet18"}:
            raise ValueError("variant must be 'compact' or 'resnet18'")
        self.variant = variant
        self.width = width
        self.embedding_dim = embedding_dim
        self.dropout = dropout

        if variant == "compact":
            self.stem = nn.Sequential(
                nn.Conv2d(1, width, 3, padding=1, bias=False),
                nn.BatchNorm2d(width),
                nn.SiLU(inplace=True),
            )
            self.features = nn.Sequential(
                DepthwiseResidualBlock(width, width),
                DepthwiseResidualBlock(width, width * 2, stride=2),
                DepthwiseResidualBlock(width * 2, width * 2),
                DepthwiseResidualBlock(width * 2, width * 4, stride=2),
                DepthwiseResidualBlock(width * 4, width * 4),
                DepthwiseResidualBlock(width * 4, width * 6, stride=2),
            )
            self.feature_layer = nn.Sequential(
                nn.Conv2d(width * 6, width * 8, 1, bias=False),
                nn.BatchNorm2d(width * 8),
                nn.SiLU(inplace=True),
            )
            feature_channels = width * 8
        else:
            try:
                from torchvision.models import ResNet18_Weights, resnet18
            except ImportError as error:
                raise ImportError("The resnet18 variant requires torchvision") from error

            weights = ResNet18_Weights.DEFAULT if pretrained else None
            backbone = resnet18(weights=weights)
            rgb_conv = backbone.conv1
            gray_conv = nn.Conv2d(
                1,
                rgb_conv.out_channels,
                kernel_size=rgb_conv.kernel_size,
                stride=rgb_conv.stride,
                padding=rgb_conv.padding,
                bias=False,
            )
            with torch.no_grad():
                gray_conv.weight.copy_(rgb_conv.weight.sum(dim=1, keepdim=True))
            self.stem = nn.Sequential(gray_conv, backbone.bn1, backbone.relu)
            self.features = nn.Sequential(backbone.layer1, backbone.layer2, backbone.layer3)
            self.feature_layer = backbone.layer4
            feature_channels = 512

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.embedding_head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(feature_channels, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.SiLU(inplace=True),
        )
        self.classifier = nn.Linear(embedding_dim, num_classes)
        self.affect_regressor = nn.Sequential(nn.Linear(embedding_dim, 2), nn.Tanh())

        if variant == "compact":
            self.apply(self._init_weights)
        else:
            self.embedding_head.apply(self._init_weights)
            self.classifier.apply(self._init_weights)
            self.affect_regressor.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
        elif isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: Tensor) -> ModelOutput:
        if x.ndim != 4 or x.shape[1] != 1:
            raise ValueError(f"Expected input shape [B, 1, H, W], received {tuple(x.shape)}")
        features = self.feature_layer(self.features(self.stem(x)))
        embedding = self.embedding_head(self.pool(features))
        return ModelOutput(
            logits=self.classifier(embedding),
            valence_arousal=self.affect_regressor(embedding),
            embedding=embedding,
        )

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
