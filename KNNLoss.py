import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

@torch.no_grad()
def _chunked_topk_cosine(
    queries: torch.Tensor,     # (B, D) normalized
    bank: torch.Tensor,        # (N, D) normalized
    k: int = 5,
    chunk_size: int = 65536
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    在大 bank 上做 chunked topk cosine：返回 topk sims 和 indices
    - queries/bank 必须已 L2 normalize
    - 返回：
        topk_sims: (B, k)
        topk_idx:  (B, k)  index in [0, N)
    """
    device = queries.device
    B, D = queries.shape
    N = bank.shape[0]

    # 初始化为很小
    topk_sims = torch.full((B, k), -1e9, device=device, dtype=queries.dtype)
    topk_idx  = torch.full((B, k), -1,   device=device, dtype=torch.long)

    base = 0
    while base < N:
        end = min(base + chunk_size, N)
        chunk = bank[base:end].to(device, non_blocking=True)  # (C, D)

        with torch.cuda.amp.autocast(enabled=False):
            sims = queries.float() @ chunk.float().t()
        sims = sims.clamp(-1.0, 1.0)

        # 当前 chunk 内 topk
        c_sims, c_idx = torch.topk(sims, k=min(k, sims.shape[1]), dim=1)
        c_idx = c_idx + base

        # 合并历史 topk 与 chunk topk
        merged_sims = torch.cat([topk_sims, c_sims], dim=1)  # (B, 2k)
        merged_idx  = torch.cat([topk_idx,  c_idx], dim=1)

        new_sims, new_pos = torch.topk(merged_sims, k=k, dim=1)
        new_idx = torch.gather(merged_idx, 1, new_pos)

        topk_sims, topk_idx = new_sims, new_idx
        base = end

    return topk_sims, topk_idx


class KNNAlignedLoss(nn.Module):
    """
    Stage2 用：对齐 kNN 推理的训练损失（只用 real 也能工作）

    组成（可独立消融）：
      1) L_dens:   kNN 平均距离（= 1 - topk cosine 的均值）
      2) L_nbr:    邻域分布一致性 KL（clean vs noisy_detach）
      3) L_var:    防塌缩 variance 正则（VICReg 风格轻量版）
      4) L_anchor: 可选，用 centroid 做一个很弱的全局锚（不是必须）

    你可以用 weights 控制每一项是否启用。
    """

    def __init__(
        self,
        k: int = 5,              # 对齐推理的 k
        m: int = 50,             # 邻域分布一致性用的候选邻居数（m>=k）
        tau: float = 0.07,       # softmax 温度
        var_gamma: float = 0.1,  # 方差下限（embedding 每维 std 至少这么大）
        chunk_size: int = 65536,
        w_dens: float = 1.0,
        w_nbr: float = 1.0,
        w_var: float = 0.05,
        w_anchor: float = 0.0,   # 先默认关掉，等你稳定后再开
    ):
        super().__init__()
        assert m >= k, "m must be >= k"
        self.k = k
        self.m = m
        self.tau = tau
        self.var_gamma = var_gamma
        self.chunk_size = chunk_size

        self.w_dens = w_dens
        self.w_nbr = w_nbr
        self.w_var = w_var
        self.w_anchor = w_anchor

    def _density_loss(self, z: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
        """
        L_dens = mean_{i} [ 1 - mean_{b in kNN(z_i)} cos(z_i, b) ]
        """
        z = F.normalize(z, p=2, dim=1)
        bank = F.normalize(bank, p=2, dim=1)

        topk_sims, _ = _chunked_topk_cosine(z, bank, k=self.k + 1, chunk_size=self.chunk_size)  # (B,k+1)
                    # ===== DEBUG PRINT (only once) =====
        if not hasattr(self, "_debug_printed"):
            self._debug_printed = True
            top1_sim = topk_sims[:, 0]  # (B,)

            print("\n[DEBUG] ---- kNN sim stats ----")
            print(f"[DEBUG] top1_sim: mean={top1_sim.mean().item():.6f}  min={top1_sim.min().item():.6f}  max={top1_sim.max().item():.6f}")
            print(f"[DEBUG] topk_sims(k={self.k}): mean={topk_sims.mean().item():.6f}  std={topk_sims.std(unbiased=False).item():.6f}")
            print("[DEBUG] ------------------------\n")
        # ==================================

        topk_sims = topk_sims[:, 1:]  # 丢掉 top1（通常是自身/近重复）
        return (1.0 - topk_sims.mean(dim=1)).mean()

    def _neighbor_kl_loss(self, z_clean: torch.Tensor, z_noisy_detach: torch.Tensor, bank: torch.Tensor) -> torch.Tensor:
        """
        邻域分布一致性：
          - 先用 clean 的 top-m 邻居做候选集 S_i
          - 对 S_i 计算 p(j|clean), q(j|noisy_detach)
          - L_nbr = mean KL(q || p)  （让 clean 去匹配 noisy 的邻域分布）
        """
        zc = F.normalize(z_clean, p=2, dim=1)
        zn = F.normalize(z_noisy_detach, p=2, dim=1)
        bank = F.normalize(bank, p=2, dim=1)

        # 用 clean 找 top-m 邻居（索引）
        topm_sims, topm_idx = _chunked_topk_cosine(zc, bank, k=self.m + 1, chunk_size=self.chunk_size)  # (B,m+1)
        topm_idx = topm_idx[:, 1:]  # 丢掉 top1
        b = bank[topm_idx]   

        # 对候选集算 clean/noisy 的相似度 logits
        # (B,m,D) * (B,1,D) -> (B,m)
        logits_c = (b * zc.unsqueeze(1)).sum(dim=-1) / self.tau
        logits_n = (b * zn.unsqueeze(1)).sum(dim=-1) / self.tau

        # p = softmax(clean), q = softmax(noisy_detach)
        log_p = F.log_softmax(logits_c, dim=1)
        q = F.softmax(logits_n, dim=1)

        # KL(q||p) = sum q (log q - log p)
        log_q = torch.log(q + 1e-12)
        kl = (q * (log_q - log_p)).sum(dim=1).mean()
        return kl

    def _variance_loss(self, z: torch.Tensor) -> torch.Tensor:
        """
        VICReg-style variance loss:
          std_d = Std(z[:,d])
          L_var = mean_d ReLU(gamma - std_d)
        """
        z = z - z.mean(dim=0, keepdim=True)
        std = torch.sqrt(z.var(dim=0, unbiased=False) + 1e-4)
        return F.relu(self.var_gamma - std).mean()

    def _anchor_loss(self, z: torch.Tensor, centroid: torch.Tensor) -> torch.Tensor:
        """
        可选：弱全局锚（防止坐标系漂移太快）
        L_anchor = mean(1 - cos(z, c))
        """
        z = F.normalize(z, p=2, dim=1)
        c = F.normalize(centroid, p=2, dim=0)
        sim = torch.matmul(z, c)
        return (1.0 - sim).mean()

    def forward(
        self,
        z_clean: torch.Tensor,
        z_noisy: torch.Tensor,
        memory_bank: torch.Tensor,
        centroid: Optional[torch.Tensor] = None,
        detach_noisy: bool = True
    ) -> Dict[str, torch.Tensor]:

        if detach_noisy:
            z_noisy = z_noisy.detach()

        losses = {}

        # 1) 密度对齐（核心）
        if self.w_dens > 0:
            L_dens = self._density_loss(z_clean, memory_bank)
            losses["loss_dens"] = L_dens
        else:
            losses["loss_dens"] = z_clean.new_tensor(0.0)

        # 2) 邻域分布一致性（增强鲁棒）
        if self.w_nbr > 0:
            L_nbr = self._neighbor_kl_loss(z_clean, z_noisy, memory_bank)
            losses["loss_nbr"] = L_nbr
        else:
            losses["loss_nbr"] = z_clean.new_tensor(0.0)

        # 3) 防塌缩
        if self.w_var > 0:
            L_var = self._variance_loss(z_clean)
            losses["loss_var"] = L_var
        else:
            losses["loss_var"] = z_clean.new_tensor(0.0)

        # 4) 可选：全局锚
        if self.w_anchor > 0 and centroid is not None:
            L_anchor = self._anchor_loss(z_clean, centroid)
            losses["loss_anchor"] = L_anchor
        else:
            losses["loss_anchor"] = z_clean.new_tensor(0.0)

        # 总损失
        total = (
            self.w_dens   * losses["loss_dens"] +
            self.w_nbr    * losses["loss_nbr"] +
            self.w_var    * losses["loss_var"] +
            self.w_anchor * losses["loss_anchor"]
        )
        losses["loss"] = total
        return losses
