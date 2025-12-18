# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
# %load_ext autoreload
# %autoreload 2

# %%
import math
import random
from argparse import Namespace
from copy import deepcopy
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from models.MultiView.Tinyvit.model import TinyVit
from models.MultiView.esresnet.fbsp import ESResNeXtFBSP
from models.MultiView.utils import Hubert_ASR
from myutils.tools import freeze_modules
from myutils.torch.nn import LambdaFunctionModule
from einops import rearrange
from torch.nn import LayerNorm

def get_global_edge_index(num_nodes=3):
    edge_index = []
    for i in range(num_nodes):
        for j in range(num_nodes):
            edge_index.append([i, j])
    return torch.tensor(edge_index, dtype=torch.long).t().contiguous()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 全局共享的edge_index（所有样本共享同一个图结构）
GLOBAL_EDGE_INDEX = get_global_edge_index().to(device)
# %%
from torchvision.transforms import v2

from myutils.zyz.EGAT import *
from myutils.zyz.my_utils import ModalityAligner
from transformers import AutoProcessor, HubertModel
# %% editable=true slideshow={"slide_type": ""}
try:
    from .rawnet.rawnet2 import RawNet2
    from .resnet import ResNet, convert_2d_to_1d
    from .resnet1d import ResNet1D
    from .utils import Expand, Squeeze, WavLM_1D, GatedFusionLayer
except ImportError:
    from rawnet.rawnet2 import RawNet2
    from resnet import ResNet, convert_2d_to_1d
    from resnet1d import ResNet1D
    from utils import Expand, Squeeze, WavLM_1D, GatedFusionLayer


# %%
# feature_model2D = ResNet(pretrained=True, verbose=True)
# feature_model1D = WavLM_1D()

# x = torch.randn(3, 1, 48000)
# x = feature_model2D.compute_stage1(x)
# x = feature_model2D.compute_stage2(x)
# x = feature_model2D.compute_stage3(x)
# x = feature_model2D.compute_stage4(x)
# x = feature_model2D.compute_latent_feature(x)
# %%
# squeeze_modules, expand_modules = [], []
# for dim, h, w in [[64, 65, 65], [128, 33, 33], [256, 17, 17], [512, 9, 9]]:
#     squeeze_modules.append(Squeeze(time_len=149, time_dim=768, spec_dim=dim, spec_height=h, spec_width=w))
#     expand_modules.append(Expand(time_len=149, time_dim=768, spec_dim=dim, spec_height=h, spec_width=w))


# %% editable=true slideshow={"slide_type": ""}
class MultiViewModel_pro(nn.Module):
    def __init__(self, verbose=0, cfg=None, args=None, **kwargs):
        super().__init__()

        self.cfg = cfg
        self.args = args
        self.graph_builder = DynamicGraphBuilder()
        self.feature_model2D = TinyVit()
        self.feature_model1D = WavLM_1D()
        # self.feature_modelASR = ESResNeXtFBSP(
        #     n_fft=2048,
        #     hop_length=561,
        #     win_length=1654,
        #     window="blackmanharris",
        #     normalized=True,
        #     onesided=True,
        #     spec_height=-1,
        #     spec_width=-1,
        #     num_classes=1024,
        #     apply_attention=True,
        #     pretrained=False,
        # )
        
        self.feature_modelASR = Hubert_ASR()
        self.GAT = EnhancedGAT_reduce()
        self.modality_aligner = ModalityAligner()
        if cfg is not None and cfg.only_1D:
            freeze_modules(self.feature_model2D)
        if cfg is not None and cfg.only_2D:
            freeze_modules(self.feature_model1D)

        use_PE = True
        use_fusion = None
        # drop_layer = cfg.drop_layer

        # self.gated_layer = GatedFusionLayer(768, 512, 768+512)

        final_dim = 768 + 512
        # final_dim = 512
        self.cls_final = nn.Sequential(
            nn.Linear(final_dim, final_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(final_dim, 1),
        )

        self.feat_dim = [768, 1024, 1024]
        self.cls1D, self.cls2D, self.clsASR = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(512, 1),
                )
                for i in range(3)
            ]
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.set_verbose(verbose)

    def norm_feat(self, feat):
        feat = feat / (1e-9 + torch.norm(feat, p=2, dim=-1, keepdim=True))
        return feat

    def print_shape(self, *args):
        for x in args:
            print(x.shape)

    def set_verbose(self, verbose):
        self.feature_model1D.verbose = verbose
        self.feature_model2D.verbose = verbose
        self.verbose = verbose

    def forward(
        self,
        x,
        stage="test",
        batch=None,
        spec_aug=None,
        freeze_feature_extractor=False,
        **kwargs
    ):
        # torch.cuda.empty_cache() 
        batch_size = x.shape[0]
        res = {}

        # with torch.no_grad():
        wav1 = self.feature_model1D.compute_stage1(x)
        spec1, res["raw_spec"] = self.feature_model2D.compute_stage1(
            x, spec_aug=spec_aug
        )
        asr1 = self.feature_modelASR.compute_stage1(x)

        if freeze_feature_extractor:
            _ = torch.set_grad_enabled(False)

        wav_feat, res["raw_wav_feat"] = self.feature_model1D.compute_rest_stage(wav1)
        # wav_feat = self.feature_model1D(x)
        spec_feat = self.feature_model2D.compute_rest_stage(spec1)
        # audio_feat = self.feature_modelESR(x)
        asr_feat = self.feature_modelASR.compute_rest_stage(asr1)
        # asr_feat = self.feature_modelASR(x.squeeze(1)).last_hidden_state.mean(1)

        if freeze_feature_extractor and self.training:
            _ = torch.set_grad_enabled(True)

        res["feature1D"] = self.norm_feat(wav_feat)
        res["feature2D"] = self.norm_feat(spec_feat)
        res["featureASR"] = self.norm_feat(asr_feat)
        res["feature"] = self.norm_feat(
            torch.concat([wav_feat, spec_feat, asr_feat], dim=-1)
        )

        res["logit1D"] = self.cls1D(res["feature1D"]).squeeze(-1)
        res["logit2D"] = self.cls2D(res["feature2D"]).squeeze(-1)
        res["logitASR"] = self.clsASR(res["featureASR"]).squeeze(-1)
        # features = self.modality_aligner(asr_feat, wav_feat, spec_feat)
        features = [asr_feat, wav_feat, spec_feat]
        batch_data = self.graph_builder.build_batch(features, GLOBAL_EDGE_INDEX)
        # batch_data = batch_data.to(device)
        # res["logit"] = (res["logit1D"] + res["logit2D"])/2
        # 前向传播
        logits, contrib_weights, gat_feat= self.GAT(batch_data.x, batch_data.edge_index, batch_data.batch)
        res["logit"] = logits
        avg_contrib = contrib_weights.mean(dim=0) 
        reciprocal_contrib = 1.0 / (avg_contrib + 1e-8)
        reciprocal_contrib_list = reciprocal_contrib.cpu().tolist()
        res["final_feat"]=gat_feat
        return res


# %% editable=true slideshow={"slide_type": ""} tags=["style-activity", "active-ipynb"]
# # cfg = Namespace(alpha=0.5, beta=0.5, only_1D=False, only_2D=False, use_fusion=True)
# # model = MultiViewModel(verbose=1, cfg=cfg, args=cfg)
# # x = torch.randn(3, 1, 48000)
# # model(x)

# %% editable=true slideshow={"slide_type": ""} tags=["active-ipynb", "style-solution"]
# # model = model.cuda()
# # x = x.cuda()
# # model(x)


# %%
def exchange_mu_std(x, y, dim=None):
    mu_x = torch.mean(x, dim=dim, keepdims=True)
    mu_y = torch.mean(y, dim=dim, keepdims=True)
    std_x = torch.std(x, dim=dim, keepdims=True)
    std_y = torch.std(y, dim=dim, keepdims=True)

    alpha = np.random.randint(50, 100) / 100
    target_mu = alpha * mu_x + (1 - alpha) * mu_y
    target_std = alpha * std_x + (1 - alpha) * std_y
    z = target_std * ((x - mu_x) / std_x) + target_mu

    # z = random_noise(z, noise_level=10)

    return z


def random_noise(x, noise_level=10):
    add_noise_level = np.random.randint(0, noise_level) / 100
    mult_noise_level = np.random.randint(0, noise_level) / 100
    z = _noise(x, add_noise_level=add_noise_level, mult_noise_level=mult_noise_level)
    return z


def _noise(x, add_noise_level=0.0, mult_noise_level=0.0):
    add_noise = 0.0
    mult_noise = 1.0
    if add_noise_level > 0.0:
        add_noise = (
            add_noise_level
            * np.random.beta(2, 5)
            * torch.FloatTensor(x.shape).normal_().to(x.device)
        )
    if mult_noise_level > 0.0:
        mult_noise = (
            mult_noise_level
            * np.random.beta(2, 5)
            * (2 * torch.FloatTensor(x.shape).uniform_() - 1).to(x.device)
            + 1
        )
    return mult_noise * x + add_noise
