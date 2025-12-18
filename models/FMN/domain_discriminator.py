import torch
import torch.nn as nn
from torch.autograd import Function

# ----------------------------------------------------
# 1. Gradient Reversal Layer (GRL)
# ----------------------------------------------------
class GradientReversalFn(Function):
    """
    梯度反转层：
    - 前向传播: y = x
    - 反向传播: grad_output = -alpha * grad_input
    """
    @staticmethod
    def forward(ctx, x, alpha):
        # ctx 用于保存状态，供 backward 使用
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        # 梯度反转
        output = grad_output.neg() * ctx.alpha
        # 返回 output 和 None (因为 alpha 是超参，不需要梯度)
        return output, None

class GradientReversalLayer(nn.Module):
    """一个封装 GRL 的 PyTorch 模块"""
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = torch.tensor(alpha, requires_grad=False)

    def forward(self, x):
        # 实际调用自定义的 Function
        return GradientReversalFn.apply(x, self.alpha)
    
# ----------------------------------------------------
# 2. Domain Discriminator
# ----------------------------------------------------
class DomainDiscriminator(nn.Module):
    """
    域判别器：用于将特征 Z (Adapter 输出) 分类到 N 个源域中的一个。
    """
    def __init__(self, in_dim, hidden_dim=256, num_domains=4):
        super().__init__()
        
        # GRL 必须放在 Discriminator 的输入端，但我们将其集成到 DFM_PL 的 forward 中
        # 这里只定义判别网络
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim), # BN 有助于稳定 DANN 训练
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_domains) # 输出是 N 个域的 logits
        )

    def forward(self, x):
        # x: (B, D) - 从 Adapter 出来的特征
        return self.net(x)