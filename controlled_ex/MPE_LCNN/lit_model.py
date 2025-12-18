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
import math

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from einops import rearrange
# -

from torchaudio.transforms import LFCC

from myutils.torch.deepfake_detection.audio import DeepfakeAudioClassification

try:
    from .lcnn import LCNN
except ImportError:
    from lcnn import LCNN


# +
# try:
#     from senet.se_resnet import se_resnet18
# except ImportError:
#     from .senet.se_resnet import se_resnet18

# model = se_resnet18()

# x = torch.randn(2, 3, 224, 224)
# model(x)
# -



# +
def Permutation_Entropy_torch_batched(x, m, t, debug=False):
    """计算批量排列熵值
    参数:
    x: 输入数据，应为二维张量，形状[batch_size, sequence_length]
    m: 嵌入维度
    t: 时间延迟
    """
    # 获取每批数据的长度
    batch_size, seq_len = x.shape
    length = seq_len - (m - 1) * t

    # 生成索引进行批量操作
    indexes = (torch.arange(length, device=x.device).unsqueeze(0) + torch.arange(m, device=x.device).unsqueeze(1) * t)
    # 使用高效的张量操作提取需要的序列
    sequences_stacked = (
        x.unsqueeze(1).expand(-1, m, -1).gather(2, indexes.expand(batch_size, -1, -1)).transpose(1, 2)
    )
    # print(sequences_stacked.shape)

    ####升序排序并获取排序后的排列索引
    S = torch.argsort(sequences_stacked, dim=2)

    
    ####对排列索引进行编码以形成唯一的序列标识
    multiplier = torch.pow(10, torch.arange(m)).to(x.device).unsqueeze(0).unsqueeze(0)
    S_encoded = (S * multiplier).sum(-1)

    # 对每一批次数据计算排列熵
    pe_batched = torch.zeros(batch_size, device=x.device)
    # with TimerContextManager(f"sort", debug=debug):
    sorted_S, _ = torch.sort(S_encoded, 1)
    org_sorted_S = sorted_S
    sorted_S = torch.concat([sorted_S, torch.zeros(batch_size, 1, device=sorted_S.device, dtype=sorted_S.dtype)], 1)
    values, counts = torch.unique_consecutive(sorted_S, return_counts=True)
    indexs = torch.nonzero(values == 0).squeeze()

    freq_list = counts / length
    freq_list = -1 * freq_list * torch.log(freq_list)
    M = np.log(np.math.factorial(m))

    # with TimerContextManager(f"unique", debug=debug):
    for i in range(batch_size):
        # _, _counts = torch.unique(S_encoded[i], return_counts=True)
        # _sorted,_ = torch.sort(S_encoded[i])
        # _, _counts=torch.unique_consecutive(sorted_S[i], return_counts=True)
        s = 0 if i == 0 else (indexs[i - 1] + 1)
        e = indexs[i]
        _counts = counts[s:e]

        # freq_list = _counts.float() / length
        # pe_batched[i] = torch.sum(-1 * freq_list * torch.log(freq_list)) / np.log(np.math.factorial(m))
        pe_batched[i] = torch.sum(freq_list[s:e]) / M

    return pe_batched


def batched_MSE_torch(signal, max_scale: int = 20, debug=False):
    # 信号形状为[batch_size, seq_len]
    batch_size, seq_len = signal.shape
    std = torch.std(signal, dim=1, keepdim=True)

    # 初始化结果列表
    result = torch.zeros(batch_size, max_scale, device=signal.device)

    # 对于向量化，我们可能需要在这个处理中保留一些循环，
    # 由于需要处理不同尺度下的变换。
    for scale in range(1, max_scale + 1):
        reshaped = signal[:, : (seq_len // scale * scale)].reshape(batch_size, -1, scale)
        signal_new = torch.mean(reshaped, dim=2)
        # with TimerContextManager(f"{scale} PE", debug=debug):
        pe = Permutation_Entropy_torch_batched(signal_new, 10, 2, debug=debug).squeeze()
        result[:, scale - 1] = pe

    return result


def batched_compute_mpe(x, debug=False):
    # 假设x形状为[batch_size, seq_len]
    signal_flu = torch.diff(x, dim=1)
    scale = 20
    mpe = batched_MSE_torch(signal_flu, scale, debug=debug)  # (batch_size, scale)
    return mpe


# +
# from ay2.torchaudio.transforms._MPE_LFCC import compute_mpe

# x = torch.randint(0, 8, (64, 1, 48000)).float().cuda()

# with TimerContextManager(debug=True):
#     for i in range(64):
#         v = compute_mpe(x[i, 0])
#         print(v)

# # %time batched_compute_mpe(x[:, 0, :], debug=False)

# +
# for i in range(30):
#     %time batched_compute_mpe(x[:, 0, :], debug=False)
# -

# # MPE

class MPE_LCNN_lit(DeepfakeAudioClassification):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = LCNN(num_class=1)
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.save_hyperparameters()

        self.lfcc = LFCC(
            n_lfcc=60,
            speckwargs={"n_fft": 400, "hop_length": 160, "center": False},
        )

    def calcuate_loss(self, batch_res, batch):
        label = batch["label"]
        loss = self.loss_fn(batch_res["logit"], label.type(torch.float32))
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0001)
        return [optimizer]

    def _shared_pred(self, batch, batch_idx):
        audio, sample_rate = batch["audio"], batch["sample_rate"]

        # audio, mpe = audio[:, :, :48000], audio[:, :, 48000:]
        lfcc = self.lfcc(audio) # (b, 1, h, w)
        mpe = batched_compute_mpe(audio[:, 0, :])
        # mpe = torch.zeros(lfcc.shape[0], 20, device=lfcc.device)

        # mpe = lfcc[:, 0, 0, :20]
        
        lfcc = rearrange(lfcc, "b 1 h w -> b 1 (h w)")
        audio = torch.concat([mpe[:, None, :], lfcc], dim=2)

        # print(audio.shape, batch["label"].shape)
        
        audio = self.lfcc(audio)
        # audio = lfcc

        # batch_out = self.model(audio).squeeze()
        feature = self.model.extract_feature(audio)
        batch_out = self.model.make_prediction(feature).squeeze(-1)
        batch_pred = (torch.sigmoid(batch_out) + 0.5).int()
        return {"logit": batch_out, "pred": batch_pred, "feature": feature}


# +
# from ay2.torchaudio.transforms import MPE_LFCC

# x = torch.randn(1, 48000).cuda()
# mpe = MPE_LFCC()

# lfcc = LFCC(
#             n_lfcc=60,
#             speckwargs={"n_fft": 400, "hop_length": 160, "center": False},
#         ).cuda()
