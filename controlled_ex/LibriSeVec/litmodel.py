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

from copy import deepcopy

from controlled_ex.LibriSeVec.v2 import RawNet2

# + tags=["active-ipynb"] editable=true slideshow={"slide_type": ""}
# from v2 import RawNet2
# -

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
    "method_classes": 6,  # number of vocoders
}
from argparse import Namespace

class LibriSeVoc_lit(DeepfakeAudioClassification):
    def __init__(self, cfg=Namespace(method_classes=9), **kwargs):
        super().__init__()
        model_config = deepcopy(RAW_NET2_CONFIG)
        model_config["method_classes"] = cfg.method_classes
        self.model = RawNet2(model_config)
        self.loss_fn1 = nn.BCEWithLogitsLoss(pos_weight=None)
        self.loss_fn2 = nn.CrossEntropyLoss()
        self.save_hyperparameters()
    
    def calcuate_loss(self, batch_res, batch):
        label = batch["label"]
        loss1 = self.loss_fn1(batch_res["logit"], label.type(torch.float32))

        vocoder_label = batch["vocoder_label"]
        loss2 = self.loss_fn2(batch_res["batch_out_vocoder"], vocoder_label)
        return (loss1 + loss2) / 2

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001)
        return [optimizer]

    def _shared_pred(self, batch, batch_idx):
        audio, sample_rate = batch["audio"], batch["sample_rate"]
        if len(audio.shape) == 3:
            audio = audio[:, 0, :]

        feature = self.model.extract_feature(audio)
        out, out2 = self.model.make_prediction(feature)
        # out, out2 = self.model(audio)
        out = out.squeeze(-1)
        batch_pred = (torch.sigmoid(out) + 0.5).int()
        return {
            "logit": out,
            "batch_out_vocoder": out2,
            "pred": batch_pred,
            "feature" : feature
        }

    # def _shared_eval_step(self, batch, batch_idx, stage="train"):
    #     batch_res = self._shared_pred(batch, batch_idx)

    #     if stage == 'test':
    #         return batch_res
        
    #     label = batch["label"]
    #     loss = self.calcuate_loss(batch_res, batch)

    #     self.log_dict(
    #         {f"{stage}-loss": loss},
    #         on_step=False,
    #         on_epoch=True,
    #         logger=True,
    #         prog_bar=True,
    #     )
    #     batch_res["loss"] = loss
    #     return batch_res

    def _shared_eval_step(
        self, batch, batch_idx, stage="train", dataloader_idx=0, *args, **kwargs
    ):
        batch_res = self._shared_pred(batch, batch_idx)

        if stage == 'test':
            return batch_res
        
        label = batch["label"]
        loss = self.calcuate_loss(batch_res, batch)

        if not isinstance(loss, dict):
            loss = {'loss' : loss}
            
        suffix = "" if dataloader_idx == 0 else f"-dl{dataloader_idx}"
        self.log_dict(
            {f"{stage}-{key}{suffix}" : loss[key] for key in loss},
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            add_dataloader_idx=False,
            batch_size = batch['label'].shape[0]
        )
        batch_res.update(loss)
        return batch_res




# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb"]
# from argparse import Namespace
# model = LibriSeVoc_lit(cfg=Namespace(method_classes=7))
#
# for key in model.state_dict():
#     print(key, model.state_dict()[key].shape)
