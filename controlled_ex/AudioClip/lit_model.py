# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

# +
import glob
import os
import sys

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import simplejpeg
import torch
import torch.nn as nn
import torchvision as tv
from IPython.display import Audio, display
from PIL import Image

from myutils.torch.deepfake_detection.audio import DeepfakeAudioClassification

# + tags=["active-ipynb", "style-activity"] editable=true slideshow={"slide_type": ""}
# from model import AudioCLIP

# + editable=true slideshow={"slide_type": ""}
from .model import AudioCLIP


# -

# ## 测试

# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "style-solution"]
# aclp = AudioCLIP(
#     pretrained="/home/ay/data/DATA/0-model_weights/AudioClip/AudioCLIP-Full-Training.pt"
# )
#
# audio = torch.randn(2, 48000)
# ((audio_features, _, _), _), _ = aclp(audio=audio)
# audio_features, audio_features.shape
# -

# ## Lit model

# + editable=true slideshow={"slide_type": ""}
class AudioClip(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = AudioCLIP(
            pretrained="/home/zyz/data/0-model_weights/AudioClip/AudioCLIP-Full-Training.pt"
        )
        self.proj = nn.Linear(1024, 1)

    def forward(self, x):
        ((audio_features, _, _), _), _ = self.model(x)
        y = self.proj(audio_features)
        return y

    def extract_feature(self, x):
        ((audio_features, _, _), _), _ = self.model(x)
        return audio_features

    def make_prediction(self, audio_features):
        y = self.proj(audio_features)
        return y


# + editable=true slideshow={"slide_type": ""}
class AudioClip_lit(DeepfakeAudioClassification):
    def __init__(self, backend="linear", **kwargs):
        super().__init__()
        self.model = AudioClip()
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.save_hyperparameters()

    def calcuate_loss(self, batch_res, batch):
        label = batch["label"]
        loss = self.loss_fn(batch_res["logit"], label.type(torch.float32))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.0001, weight_decay=0.0001
        )
        return [optimizer]
    def forward(self, x):
        (features, logits), loss = self.model.model(x)

        return features[0]
