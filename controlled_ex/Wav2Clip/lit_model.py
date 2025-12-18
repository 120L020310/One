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

import pytorch_lightning as pl
import torch
import torch.nn as nn

from myutils.torch.deepfake_detection.audio import DeepfakeAudioClassification

# + tags=["style-activity", "active-ipynb"] editable=true slideshow={"slide_type": ""}
# from wav2clip import get_model
# -

from .wav2clip import get_model


# ## 测试

# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "style-solution"]
# model = get_model()
# x = torch.randn(2, 48000)
# model(x)
# -

# ## Lit model

class Wav2Clip(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = get_model()
        self.proj = nn.Linear(512, 1)
        
    
    def forward(self, x):
        feat = self.model(x)
        return self.proj(feat), feat


class Wav2Clip_lit(DeepfakeAudioClassification):
    def __init__(self, backend="linear", **kwargs):
        super().__init__()
        self.model = Wav2Clip()
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

    def _shared_pred(self, batch, batch_idx, stage='train', **kwargs):
        audio, sample_rate = batch["audio"], batch["sample_rate"]
        if len(audio.shape) == 3:
            audio = audio[:, 0, :]

        out, feat = self.model(audio)
        out = out.squeeze(-1)
        batch_pred = (torch.sigmoid(out) + 0.5).int()
        return {
            "logit": out,
            "pred": batch_pred,
            "feature": feat
        }
    def forward(self, x):
        feat = self.model.model(x)
        return feat
