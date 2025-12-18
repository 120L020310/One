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
from myutils.torch.deepfake_detection import DeepfakeAudioClassification

from torchaudio.transforms import LFCC

from controlled_ex.Hubert.model import Hubert_ASR


class Hubert_lit(DeepfakeAudioClassification):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = Hubert_ASR()
        self.loss_fn = nn.BCEWithLogitsLoss()


    def calcuate_loss(self, batch_res, batch):
        label = batch["label"]
        loss = self.loss_fn(batch_res["logit"], label.type(torch.float32))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=1e-6, weight_decay=1e-4
        )
        return [optimizer]

    def _shared_pred(self, batch, batch_idx):
        audio, sample_rate = batch["audio"], batch["sample_rate"]

        # asr1 = self.model.compute_stage1(audio)
        batch_out = self.model(audio)

        batch_pred = (torch.sigmoid(batch_out) + 0.5).int()
        return {"logit": batch_out, "pred": batch_pred}
    def forward(self,x):
        feat = self.model(x)
        return {"final_feat":feat}
