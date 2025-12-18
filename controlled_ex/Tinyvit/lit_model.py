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
from controlled_ex.Tinyvit.model import TinyVit
from myutils.torch.deepfake_detection import DeepfakeAudioClassification

from torchaudio.transforms import LFCC



class TinyVit_lit(DeepfakeAudioClassification):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = TinyVit()
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.cls = nn.Sequential(
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(512, 1),
                )

    def calcuate_loss(self, batch_res, batch):
        label = batch["label"]
        loss = self.loss_fn(batch_res["logit"], label.type(torch.float32))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=1e-6, weight_decay=1e-4
        )
        return [optimizer]
    
    def norm_feat(self, feat):
        feat = feat / (1e-9 + torch.norm(feat, p=2, dim=-1, keepdim=True))
        return feat

    def _shared_pred(self, batch, batch_idx, stage):
        audio, sample_rate = batch["audio"], batch["sample_rate"]

        spec1, raw_spec = self.model.compute_stage1(
            audio,stage
        )
        spec_feat = self.model.compute_rest_stage(spec1)
        feature2D = self.norm_feat(spec_feat)
        batch_out = self.cls(feature2D).squeeze(-1)
        batch_pred = (torch.sigmoid(batch_out) + 0.5).int()
        return {"logit": batch_out, "pred": batch_pred}
    def forward(self, audio):
        spec1, raw_spec = self.model.compute_stage1(
            audio,stage="test"
        )
        spec_feat = self.model.compute_rest_stage(spec1)
        feature2D = self.norm_feat(spec_feat)
        # batch_out = self.cls(feature2D).squeeze(-1)
        # batch_pred = (torch.sigmoid(batch_out) + 0.5).int()
        return feature2D,spec_feat

