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
from copy import deepcopy

from myutils.torch.deepfake_detection.audio import DeepfakeAudioClassification

from .v2 import RawNet2

# The default config of RawNet2 is
# ```python
# RAW_NET2_CONFIG  = {
#     'first_conv': 251,
#     'in_channels': 1,
#     'filts': [128, [128,128], [128,256], [256,256]],
#     'blocks': [2, 4],
#     'nb_fc_att_node': [1],
#     'nb_fc_node': 1024,
#     'gru_node': 1024,
#     'nb_gru_layer': 1,
#     'nb_samp': 59049,
#     # 'nb_classes': 1
# }
# ```
# The nb_classes denote the final prediction shape. In the code of WaveFake, the config of RawNet2 is:
# ```python
# RAW_NET2_CONFIG = {
#     "nb_samp": 64600,
#     "first_conv": 1024,   # no. of filter coefficients
#     "in_channels": 1,  # no. of filters channel in residual blocks
#     "filts": [20, [20, 20], [20, 128], [128, 128]],
#     "blocks": [2, 4],
#     "nb_fc_node": 1024,
#     "gru_node": 1024,
#     "nb_gru_layer": 3,
#     "nb_classes": 1,
# }
# ```

RAW_NET2_CONFIG = {
    "nb_samp": 48000,
    "first_conv": 1024,  # no. of filter coefficients
    "in_channels": 1,  # no. of filters channel in residual blocks
    "filts": [20, [20, 20], [20, 128], [128, 128]],
    "blocks": [2, 4],
    "nb_fc_node": 1024,
    "gru_node": 1024,
    "nb_gru_layer": 3,
    "nb_classes": 1,
}


class RawNet2_lit(DeepfakeAudioClassification):
    def __init__(self, **kwargs):
        super().__init__()
        model_config = deepcopy(RAW_NET2_CONFIG)
        self.model = RawNet2(model_config)
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=None)
        self.save_hyperparameters()
    
    def calcuate_loss(self, batch_res, batch):
        label = batch["label"]
        loss = self.loss_fn(batch_res["logit"], label.type(torch.float32))
        return loss

    def forward(self, audio):
        if len(audio.shape) == 3:
            audio = audio[:, 0, :]
        out = self.model(audio)
        out = out.squeeze(-1)
        return out
    
    def _shared_pred(self, batch, batch_idx):
        audio, sample_rate = batch["audio"], batch["sample_rate"]
        if len(audio.shape) == 3:
            audio = audio[:, 0, :]

        # out = self.model(audio)
        feature = self.model.extract_feature(audio)
        out = self.model.make_prediction(feature)
        out = out.squeeze(-1)
        batch_pred = (torch.sigmoid(out) + 0.5).int()
        return {
            "logit": out,
            "pred": batch_pred,
            "feature" : feature
        }
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.0001, weight_decay=0.0001
        )
        return [optimizer]


