"""Shared label definitions and affect-space prototypes."""

from __future__ import annotations

import torch

EMOTIONS = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "sad",
    "surprise",
    "neutral",
)

# Approximate valence/arousal prototypes on [-1, 1]. They are auxiliary targets,
# not psychological ground truth. Rows follow the FER2013 label order above.
VALENCE_AROUSAL = torch.tensor(
    [
        [-0.70, 0.70],  # angry
        [-0.60, 0.35],  # disgust
        [-0.75, 0.80],  # fear
        [0.85, 0.55],   # happy
        [-0.75, -0.35], # sad
        [0.35, 0.90],   # surprise
        [0.00, 0.00],   # neutral
    ],
    dtype=torch.float32,
)

EMOTION_TO_INDEX = {name: index for index, name in enumerate(EMOTIONS)}

