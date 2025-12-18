import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import sys
import torch
import torch.nn as nn
from torch.nn import LayerNorm
from collections import OrderedDict
import torch.nn.functional as F
from loguru import logger
import torch.optim as optim

from myutils.zyz.loss_fn import ContrastiveLoss


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(
            OrderedDict(
                [
                    ("c_fc", nn.Linear(d_model, d_model * 4)),
                    ("gelu", QuickGELU()),
                    ("c_proj", nn.Linear(d_model * 4, d_model)),
                ]
            )
        )
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = (
            self.attn_mask.to(dtype=x.dtype, device=x.device)
            if self.attn_mask is not None
            else None
        )
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(
        self, width: int, layers: int, heads: int, attn_mask: torch.Tensor = None
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(
            *[ResidualAttentionBlock(width, heads, attn_mask) for _ in range(layers)]
        )

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class ModalityAligner(nn.Module):
    def __init__(
        self,
        audio_dim=1024,
        asr_dim=768,
        spec_dim=576,
        hidden_dim=512,
        num_heads=4,
        num_layers=2,
    ):
        super(ModalityAligner, self).__init__()

        # Audio 模态（1D特征）处理
        self.audio_fc = nn.Linear(audio_dim, hidden_dim)

        # ASR 模态（3D特征）处理
        self.asr_fc = nn.Linear(asr_dim, hidden_dim)
        # self.asr_transformer = nn.TransformerEncoder(
        #     nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads), num_layers=num_layers
        # )
        self.spec_fc = nn.Linear(spec_dim, hidden_dim)
        # Spectrogram 模态（1D特征）处理
        # self.spec_fc = nn.Conv1d(spec_dim, hidden_dim, kernel_size=3, padding=1)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.fc_output = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, audio_features, asr_features, spec_features):
        # 对每个模态进行处理
        audio_output = self.audio_fc(audio_features)  # (64, 1024) -> (64, hidden_dim)
        audio_output = audio_output / (
            1e-9 + torch.norm(audio_output, p=2, dim=-1, keepdim=True)
        )
        asr_features = asr_features.to(self.device)
        asr_output = self.asr_fc(
            asr_features
        )  # (64, 149, 768) -> (64, 149, hidden_dim)
        asr_output = asr_output / (
            1e-9 + torch.norm(asr_output, p=2, dim=-1, keepdim=True)
        )
        # asr_output = asr_output.permute(1, 0, 2)  # (149, 64, hidden_dim) -> (seq_len, batch_size, hidden_dim)
        # asr_output = self.asr_transformer(asr_output)  # Transformer 对齐
        # asr_output = asr_output.mean(dim=0)  # 平均池化
        spec_features = self.spec_fc(spec_features)
        spec_output = spec_features / (
            1e-9 + torch.norm(spec_features, p=2, dim=-1, keepdim=True)
        )

        # # 融合所有模态
        # combined_output = audio_output + asr_output + spec_output
        # combined_output = F.relu(self.fc_output(combined_output))
        return audio_output, asr_output, spec_output


def gcn_triplet_loss(xa, xb, xc, xa_neg, xb_neg, xc_neg, margin=0.5):
    """基于 GCN 融合的 Triplet Loss"""

    # 计算真实音频的模态间距离
    d_ab = F.pairwise_distance(xa, xb, p=2)
    d_bc = F.pairwise_distance(xb, xc, p=2)
    d_ca = F.pairwise_distance(xc, xa, p=2)
    d_pos = (d_ab + d_bc + d_ca) / 3

    # 计算伪造音频的模态间距离（取最大值）
    d_ab_neg = F.pairwise_distance(xa_neg, xb_neg, p=2)
    d_bc_neg = F.pairwise_distance(xb_neg, xc_neg, p=2)
    d_ca_neg = F.pairwise_distance(xc_neg, xa_neg, p=2)
    d_neg = torch.max(torch.stack([d_ab_neg, d_bc_neg, d_ca_neg]))

    # 计算 Triplet Loss
    loss = torch.clamp(d_pos - d_neg + margin, min=0.0).mean()
    return loss


def mutual_information_loss(xa, xb, xc):
    def compute_mi(x1, x2):
        joint = torch.cat([x1, x2], dim=-1)
        mi = -torch.mean(torch.log(torch.sigmoid(joint)))  # 负互信息损失
        return mi

    loss_mi = compute_mi(xa, xb) + compute_mi(xb, xc) + compute_mi(xc, xa)
    return loss_mi / 3  # 取平均


def compute_loss(xa, xb, xc, xa_neg, xb_neg, xc_neg, modality_aligner):
    # 1. Transformer 进行模态对齐
    xa_aligned, xb_aligned, xc_aligned = modality_aligner(xa, xb, xc)
    xa_neg_aligned, xb_neg_aligned, xc_neg_aligned = modality_aligner(
        xa_neg, xb_neg, xc_neg
    )

    # 2. 计算 Triplet Loss
    triplet_loss = gcn_triplet_loss(
        xa_aligned,
        xb_aligned,
        xc_aligned,
        xa_neg_aligned,
        xb_neg_aligned,
        xc_neg_aligned,
    )

    # 3. 计算互信息损失
    mi_loss = mutual_information_loss(xa_aligned, xb_aligned, xc_aligned)

    # 5. 总损失
    total_loss = triplet_loss + 0.1 * mi_loss

    return total_loss


import torch
from sklearn.metrics import roc_auc_score, accuracy_score
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

def compute_binary_metrics(labels, probs):
    """计算 BinaryACC、BinaryAUC、EER"""
    # 确保数据在CPU且转为numpy
    labels = labels.cpu().numpy()
    probs = probs.cpu().numpy()
    
    # Binary Accuracy
    preds = (probs >= 0.5).astype(int)
    acc = accuracy_score(labels, preds)
    
    # Binary AUC
    auc = roc_auc_score(labels, probs)
    
    # Equal Error Rate (EER)
    fpr, tpr, thresholds = roc_curve(labels, probs)
    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    
    return acc, auc, eer

class LabelSmoothingBCE(nn.Module):

    def __init__(self, label_smoothing):
        super().__init__()
        self.label_smoothing = label_smoothing
        print("BCE loss with label smoothing: ", self.label_smoothing)

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: (B, 1)
            y_true: (B,)
        """
        assert y_true.ndim == 1
        if self.label_smoothing != 0:
            y_true = (
                y_true.float() * (1 - self.label_smoothing) + self.label_smoothing / 2
            )
        if y_pred.ndim == 2:
            y_pred = y_pred.squeeze(1)
        return F.binary_cross_entropy_with_logits(y_pred, y_true.float())
    

class post_process(nn.Module):
    def __init__(self,hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(hidden_dim, 512)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)

    def forward(self, x):  ## (batch, time, channel) --> (batch, channel)
        x = x.unsqueeze(1)
        x = self.first_bn(x)
        x = self.selu(x)
        x = F.max_pool2d(x, (3, 3))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        feat = self.selu(x)
        return feat