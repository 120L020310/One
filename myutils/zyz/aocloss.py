import torch
import torch.nn as nn
from torch.autograd.function import Function
import torch.nn.functional as F
from torch.autograd import Variable
import math
class AOCloss(nn.Module):
    '''
    AOC loss function.
    based on the paper:
    Adaptive Centroid Shift Loss (AOCloss) method for Audio Deepfake Detection
    '''
    def __init__(self, embedding_dim=2):
        super(AOCloss, self).__init__()
        self.embedding_dim = embedding_dim
        self.centroid = None
        self.n = 0  # Total number of bonafide samples encountered

    def update_centroid(self, bonafide_embeddings):
        s = bonafide_embeddings.shape[0]
        if s == 0:
            return

        Ei = bonafide_embeddings.mean(dim=0).detach()  # Detach to avoid graph tracking

        if self.centroid is None:
            self.centroid = Ei
            self.n = s
        else:
            self.centroid = ((self.n * self.centroid.detach()) + (s * Ei)) / (self.n + s)
            self.n += s

    def one_class_loss(self, bonafide_embeddings, fake_embeddings):
        if self.centroid is None:
            raise ValueError("Centroid has not been initialized with bonafide samples.")

        # Normalize embeddings and centroid
        centroid_norm = F.normalize(self.centroid.detach(), p=2, dim=0)
        bonafide_norm = F.normalize(bonafide_embeddings, p=2, dim=1)
        fake_norm = F.normalize(fake_embeddings, p=2, dim=1)

        # Compute cosine similarity
        bonafide_similarity = torch.matmul(bonafide_norm, centroid_norm)
        fake_similarity = torch.matmul(fake_norm, centroid_norm)

        Mb = bonafide_embeddings.shape[0]
        Ms = fake_embeddings.shape[0]

        if Mb == 0 :
            loc = 1 + (torch.sum(fake_similarity) / Ms)
        elif Ms ==0:
            loc = 1 + (-torch.sum(bonafide_similarity) / Mb)
            # raise ValueError("Both bonafide and spoof samples must be present in the batch.")
        else:
            loc = 1 + (-torch.sum(bonafide_similarity) / Mb) + (torch.sum(fake_similarity) / Ms)
        return loc

    def forward(self, embeddings, labels=None,stage = "train"):
        bonafide_embeddings = embeddings[labels == 1]
        fake_embeddings = embeddings[labels == 0]
        # if stage =="train":
        self.update_centroid(bonafide_embeddings)
        loss = self.one_class_loss(bonafide_embeddings, fake_embeddings)
        return loss

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class AOCloss_plus(nn.Module):
    """
    AOCloss (real-only) + global gating (Fix #1)
    - inliers: update centroid
    - outliers: do NOT update centroid
    - LOSS: all bonafide are pulled toward centroid, but outliers get smaller weight (robust)
    
    IMPORTANT:
    - centroid / 统计量不是 register_buffer/Parameter -> 不会自动保存到 state_dict()
      若要 test / 下个 epoch / resume 继续使用同一 centroid/统计量：
        保存 checkpoint 时额外保存 export_aux_state()
        加载 checkpoint 后调用 load_aux_state(...)
    """
    def __init__(self, embedding_dim=2, gate_factor=4.0, warmup_count=32, w_out=0.2, eps=1e-12):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.gate_factor = gate_factor
        self.warmup_count = warmup_count
        self.w_out = w_out
        self.eps = eps

        # ---- state (普通变量，不会进 state_dict) ----
        self.centroid = None  # torch.Tensor [D] on correct device

        self.n_used = 0
        self.dist_sum = 0.0
        self.dist_count = 0

        # for logging/monitoring
        self.feature_stability = None
        self.last_keep_ratio = None
        self.last_thr = None
        self.just_init = False

    # ------------------ (可选) 手动保存/恢复这些状态 ------------------
    def export_aux_state(self):
        return {
            "centroid": None if self.centroid is None else self.centroid.detach().cpu(),
            "n_used": int(self.n_used),
            "dist_sum": float(self.dist_sum),
            "dist_count": int(self.dist_count),
        }

    def load_aux_state(self, state: dict, device=None):
        c = state.get("centroid", None)
        if c is None:
            self.centroid = None
        else:
            self.centroid = c.to(device) if device is not None else c

        self.n_used = int(state.get("n_used", 0))
        self.dist_sum = float(state.get("dist_sum", 0.0))
        self.dist_count = int(state.get("dist_count", 0))

    # ------------------ core logic ------------------
    @torch.no_grad()
    def _ensure_centroid_device(self, device):
        if self.centroid is not None and self.centroid.device != device:
            self.centroid = self.centroid.to(device)

    @torch.no_grad()
    def _init_centroid(self, bonafide_embeddings: torch.Tensor):
        self.centroid = bonafide_embeddings.mean(dim=0).detach()
        s = bonafide_embeddings.shape[0]
        self.n_used += s

        c_norm = F.normalize(self.centroid, p=2, dim=0)
        b_norm = F.normalize(bonafide_embeddings.detach(), p=2, dim=1)
        dist = 1.0 - (b_norm @ c_norm)

        self.dist_sum += float(dist.sum().item())
        self.dist_count += int(dist.numel())

        mean_dist = self.dist_sum / max(1, self.dist_count)
        self.feature_stability = float(1.0 / (mean_dist + self.eps))
        self.last_keep_ratio = 1.0
        self.last_thr = None
        self.just_init = True

    def split_by_gate(self, bonafide_embeddings: torch.Tensor):
        """
        用过去 inlier 的均值距离做阈值：
          dist > gate_factor * mean_dist_past -> outlier
        返回: inliers, outliers
        """
        s = bonafide_embeddings.shape[0]
        if s == 0:
            return bonafide_embeddings, bonafide_embeddings[:0]

        if self.centroid is None:
            return bonafide_embeddings, bonafide_embeddings[:0]

        device = bonafide_embeddings.device
        self._ensure_centroid_device(device)

        # 先在 no_grad 里算 dist/阈值/mask + 更新统计（不需要梯度）
        with torch.no_grad():
            mean_dist_past = None
            if self.dist_count >= 1:
                mean_dist_past = self.dist_sum / self.dist_count

            c_norm = F.normalize(self.centroid, p=2, dim=0)
            b_norm = F.normalize(bonafide_embeddings.detach(), p=2, dim=1)
            dist = 1.0 - (b_norm @ c_norm)  # [s]

            if (mean_dist_past is None) or (self.dist_count < self.warmup_count):
                keep_mask = torch.ones_like(dist, dtype=torch.bool)
                thr = None
            else:
                thr = self.gate_factor * mean_dist_past
                keep_mask = dist <= thr

            self.last_keep_ratio = float(keep_mask.float().mean().item())
            self.last_thr = thr

            # 只用 inlier 的距离更新统计（让阈值代表“正常范围”）
            if not self.just_init:
                kept_dist = dist[keep_mask]
                self.dist_sum += float(kept_dist.sum().item())
                self.dist_count += int(kept_dist.numel())
            self.just_init = False

            mean_dist_all = self.dist_sum / max(1, self.dist_count)
            self.feature_stability = float(1.0 / (mean_dist_all + self.eps))

        # 关键：mask 索引要在 no_grad 外，这样 inliers/outliers 仍保留梯度
        inliers = bonafide_embeddings[keep_mask]
        outliers = bonafide_embeddings[~keep_mask]
        return inliers, outliers

    @torch.no_grad()
    def update_centroid_with_inliers(self, inliers: torch.Tensor):
        k = inliers.shape[0]
        if k == 0:
            return
        self.centroid = (self.centroid * float(self.n_used) + inliers.detach().sum(dim=0)) / float(self.n_used + k)
        self.n_used += int(k)

    def one_class_loss(self, inliers: torch.Tensor, outliers: torch.Tensor):
        """
        Fix #1：不把 outliers 当 fake。
        所有 bonafide 都拉近 centroid，但 outliers 权重更小（w_out）。
          loss = 1 - (sum(sim_in) + w_out*sum(sim_out)) / (Mb + w_out*Ms)
        """
        if self.centroid is None:
            raise ValueError("Centroid has not been initialized with bonafide samples.")

        device = inliers.device if inliers.numel() > 0 else outliers.device
        self._ensure_centroid_device(device)
        centroid_norm = F.normalize(self.centroid.detach(), p=2, dim=0)

        Mb = inliers.shape[0]
        Ms = outliers.shape[0]

        # 如果这步真的发生（极少），返回一个“挂钩”的 0，避免 backward 报错
        if Mb == 0 and Ms == 0:
            return (inliers.sum() + outliers.sum()) * 0.0

        num = 0.0
        den = 0.0

        if Mb > 0:
            sim_in = F.normalize(inliers, p=2, dim=1) @ centroid_norm
            num = num + sim_in.sum()
            den = den + float(Mb)

        if Ms > 0:
            sim_out = F.normalize(outliers, p=2, dim=1) @ centroid_norm
            num = num + (self.w_out * sim_out.sum())
            den = den + (self.w_out * float(Ms))

        # 防止除 0
        den = max(den, self.eps)

        # maximize weighted mean similarity => minimize (1 - mean_sim)
        loc = 1.0 - (num / den)
        return loc

    def forward(self, embeddings, labels=None, stage="train"):
        bonafide_embeddings = embeddings[labels == 1]

        if bonafide_embeddings.shape[0] > 0 and (self.centroid is None):
            self._init_centroid(bonafide_embeddings)

        inliers, outliers = self.split_by_gate(bonafide_embeddings)

        if self.centroid is not None:
            self.update_centroid_with_inliers(inliers)

        loss = self.one_class_loss(inliers, outliers)
        return loss


import torch
import torch.nn as nn
import torch.nn.functional as F

class AOCloss_plus_soft(nn.Module):
    """
    AOCloss (real-only) + soft weighting by distance to centroid
    - No hard outlier split
    - Weight w_i decreases as sample is farther from centroid

    IMPORTANT:
    - centroid / 统计量不是 register_buffer/Parameter -> 不会自动保存到 state_dict()
      若要 test / 下个 epoch / resume 继续使用同一 centroid/统计量：
        保存 checkpoint 时额外保存 export_aux_state()
        加载 checkpoint 后调用 load_aux_state(...)
    """
    def __init__(
        self,
        embedding_dim=2,
        base_margin=0.01,   # s0 = min(sim) - base_margin；你说的“最低值-1%”
        weight_power=1.0,   # 可选：>1 会更强调近样本（类似 focal），先用 1.0
        eps=1e-12
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.base_margin = base_margin
        self.weight_power = weight_power
        self.eps = eps

        # ---- state (普通变量，不会进 state_dict) ----
        self.centroid = None  # torch.Tensor [D]

        # 用“权重总和”当作有效样本量（比 int 计数更符合加权更新）
        self.weight_used = 0.0

        # for logging/monitoring
        self.last_base = None
        self.last_w_mean = None
        self.last_w_min = None
        self.last_w_max = None
        self.last_sim_mean = None
        self.last_sim_min = None

    # ------------------ (可选) 手动保存/恢复这些状态 ------------------
    def export_aux_state(self):
        return {
            "centroid": None if self.centroid is None else self.centroid.detach().cpu(),
            "weight_used": float(self.weight_used),
        }

    def load_aux_state(self, state: dict, device=None):
        c = state.get("centroid", None)
        if c is None:
            self.centroid = None
        else:
            self.centroid = c.to(device) if device is not None else c
        self.weight_used = float(state.get("weight_used", 0.0))

    @torch.no_grad()
    def _ensure_centroid_device(self, device):
        if self.centroid is not None and self.centroid.device != device:
            self.centroid = self.centroid.to(device)

    @torch.no_grad()
    def _init_centroid(self, bonafide_embeddings: torch.Tensor):
        # 初始化：直接用 batch mean
        self.centroid = bonafide_embeddings.mean(dim=0).detach()
        # 初始化时认为“权重使用量”≈样本数（你也可以设成 0.0，但这样更平滑）
        self.weight_used += float(bonafide_embeddings.shape[0])

    def _compute_similarity(self, embeddings: torch.Tensor):
        """
        embeddings: [B, D]
        return sim: [B] cosine similarity to centroid
        """
        centroid_norm = F.normalize(self.centroid, p=2, dim=0)          # [D]
        emb_norm = F.normalize(embeddings, p=2, dim=1)                  # [B, D]
        sim = emb_norm @ centroid_norm                                  # [B]
        return sim

    def _compute_weights(self, sim_detach: torch.Tensor):
        base = float(sim_detach.min().item()) - float(self.base_margin)
        base = max(-1.0, min(1.0, base))

        w = torch.relu(sim_detach - base)
        if self.weight_power != 1.0:
            w = w.pow(self.weight_power)

        # 如果全 0，退化为均匀权重
        if float(w.sum().item()) <= self.eps:
            raise KeyError

        # ✅ 关键：缩放，使得 sum(w) == batch_size（等价 mean(w)=1）
        B = sim_detach.numel()
        w = w * (B / (w.sum() + self.eps))

        return w, base


    @torch.no_grad()
    def update_centroid_weighted(self, embeddings: torch.Tensor, w_detach: torch.Tensor):
        """
        用权重做加权 running mean 更新 centroid：
          centroid <- (W_old * centroid + sum(w_i * x_i)) / (W_old + sum(w_i))
        """
        W_new = float(w_detach.sum().item())
        if W_new <= self.eps:
            raise KeyError

        # 注意：这里用原始 embeddings（不 normalize），和你之前 update centroid 一致
        num = self.centroid * float(self.weight_used) + (embeddings.detach() * w_detach.unsqueeze(1)).sum(dim=0)
        den = float(self.weight_used) + W_new
        self.centroid = num / max(den, self.eps)
        self.weight_used = den

    def one_class_loss_weighted(self, embeddings: torch.Tensor, w_detach: torch.Tensor):
        """
        loss = 1 - weighted_mean(sim)
        """
        sim = self._compute_similarity(embeddings)  # 带梯度
        w = w_detach                                # 无梯度

        # 记录日志（可选）
        with torch.no_grad():
            self.last_sim_mean = float(sim.mean().item())
            self.last_sim_min = float(sim.min().item())
            self.last_w_mean = float(w.mean().item())
            self.last_w_min = float(w.min().item())
            self.last_w_max = float(w.max().item())

        weighted_mean = (w * sim).sum() / (w.sum() + self.eps)
        loss = 1.0 - weighted_mean
        return loss

    def forward(self, embeddings, labels=None, stage="train"):
        bonafide_embeddings = embeddings[labels == 1]
        if len(bonafide_embeddings)==0:return 0
        if self.centroid is None:
            self._init_centroid(bonafide_embeddings)

        device = bonafide_embeddings.device
        self._ensure_centroid_device(device)

        # 1) 用 no_grad 计算“用于加权的相似度”和权重
        with torch.no_grad():
            sim_detach = self._compute_similarity(bonafide_embeddings).detach()
            w_detach, base = self._compute_weights(sim_detach)
            self.last_base = base

        # 2) 用权重更新 centroid（no_grad）
        self.update_centroid_weighted(bonafide_embeddings, w_detach)

        # 3) 用同样权重计算 loss（有梯度）
        loss = self.one_class_loss_weighted(bonafide_embeddings, w_detach)
        return loss
