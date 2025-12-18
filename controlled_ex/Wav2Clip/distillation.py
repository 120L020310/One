# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

# +
import json
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from tqdm import tqdm
# -

import pytorch_lightning as pl
from ay2.torch.deepfake_detection import DeepfakeAudioClassification
from ay2.torch.lightning.callbacks import (
    ACC_Callback,
    APCallback,
    AUC_Callback,
    Color_progress_bar,
    EarlyStop,
    EER_Callback,
)
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from ._data import get_data
from .wav2clip import get_model

# + editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "style-activity"]
# from wav2clip import get_model

# + editable=true slideshow={"slide_type": ""}
from models.Ours.model import AudioModel


# -

# ### Loss function

class CLIPLoss1D(nn.Module):
    def __init__(self):
        super(CLIPLoss1D, self).__init__()
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.loss_image = nn.CrossEntropyLoss()
        self.loss_text = nn.CrossEntropyLoss()

    def forward(self, image_features, text_features):
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logit_scale * text_features @ image_features.t()

        batch_size = image_features.shape[0]
        ground_truth = torch.arange(
            batch_size, dtype=torch.long, device=image_features.device
        )
        return (
            self.loss_image(logits_per_image, ground_truth)
            + self.loss_text(logits_per_text, ground_truth)
        ) / 2


# + editable=true slideshow={"slide_type": ""}
class Our_lit(DeepfakeAudioClassification):
    def __init__(self, num_classes=1, **kwargs):
        super().__init__()
        self.model1 = get_model()
        self.model2 = AudioModel().feature_model
        dim = self.model2.dims[-1]
        self.mlp1 = nn.Sequential(
                    nn.Linear(512, 512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1),
                    nn.Linear(512, dim),
                )
        self.mlp2 = nn.Sequential(
                    nn.Linear(dim, 512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.1),
                    nn.Linear(512, 512),
                )

        self.drop = nn.Dropout(0.1)
        self.class_head1 = nn.Linear(512, 309)
        self.class_head2 = nn.Linear(dim, 309)

        self.loss_fn = CLIPLoss1D()
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.save_hyperparameters()

    def calcuate_loss(self, batch_res, batch):
        res = {}
        res["loss1"] = self.loss_fn(
            self.mlp1(batch_res["feat_org"]), batch_res["feat_tar"]
        )
        res["loss2"] = self.loss_fn(
            batch_res["feat_org"], self.mlp2(batch_res["feat_tar"])
        )
        res["ce_loss1"] = self.ce_loss(batch_res["org_logit"], batch["label"])
        res["ce_loss2"] = self.ce_loss(batch_res["tar_logit"], batch["label"])
        # return (loss1 + loss2) / 2
        self.log_dict(res, on_epoch=True, prog_bar=True, logger=True)
        return (res["loss1"] + res["loss2"] + res["ce_loss1"] + res["ce_loss2"]) / 4

    def configure_optimizers(self):
        import itertools
        optimizer = torch.optim.Adam(
            itertools.chain(
                self.model1.parameters(),
                self.model2.parameters(),
                self.mlp1.parameters(),
                self.mlp2.parameters(),
                self.class_head1.parameters(),
                self.class_head2.parameters(),
            ),
            lr=0.0001, weight_decay=0.0001
        )
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.8,
            patience=3,
            threshold=0.001,
            threshold_mode="abs",
            min_lr=0.000001,
            eps=1e-08,
            verbose=True,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val-loss",
            },
        }

    def _shared_pred(self, batch, batch_idx):
        audio, sample_rate = batch["audio"], batch["sample_rate"]
        # if len(audio.shape) == 3:
        # audio = audio[:, 0, :]

        # with torch.no_grad():
        feature_org = self.model1(audio[:, 0, :])

        feature_tar = self.model2(audio)

        res = {"feat_org": feature_org, "feat_tar": feature_tar}
        res["org_logit"] = self.class_head1(self.drop(feature_org))
        res["tar_logit"] = self.class_head2(self.drop(feature_tar))

        return res


# -

def make_callbacks():
    callbacks = [
        Color_progress_bar(),
        ModelCheckpoint(
            dirpath=None,
            save_top_k=1,
            monitor="val-loss",
            mode="min",
            save_last=True,
            filename="best-{epoch}-{val-loss:.2f}",
        ),
        EarlyStop(
            min_epochs=50,
            monitor="val-loss",
            min_delta=0.001,
            patience=5,
            mode="min",
            stopping_threshold=0.1,
            verbose=False,
        ),
        ACC_Callback(
            batch_key="label", output_key="org_logit", num_classes=309, theme="org"
        ),
        AUC_Callback(
            batch_key="label", output_key="org_logit", num_classes=309, theme="org"
        ),
        ACC_Callback(
            batch_key="label", output_key="tar_logit", num_classes=309, theme="tar"
        ),
        AUC_Callback(
            batch_key="label", output_key="tar_logit", num_classes=309, theme="tar"
        ),
    ]
    return callbacks


# +
ROOT_DIR = "/home/ay/data/DATA/1-model_save/0-Audio"


def start_distillation(gpu):
    dl = get_data()

    trainer = pl.Trainer(
        max_epochs=300,
        accelerator="gpu",
        devices=[gpu],
        logger=pl.loggers.CSVLogger(
            ROOT_DIR,
            name="distillation",
            version=None,
        ),
        check_val_every_n_epoch=1,
        callbacks=make_callbacks(),
        default_root_dir=ROOT_DIR,
    )
    model = Our_lit()

    trainer.fit(model, dl.train, val_dataloaders=dl.val)

    torch.save(model.model2.state_dict(), trainer.logger.log_dir + "/model2.ckpt")
