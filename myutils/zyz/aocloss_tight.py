import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class AOCloss_Tight(nn.Module):
    '''
    AOCloss with Temperature Scaling (Log-Sum-Exp).
    This version forces tighter clustering by penalizing distant bonafide samples exponentially.
    '''
    def __init__(self, embedding_dim=256, temperature=0.001):
        super(AOCloss_Tight, self).__init__()
        self.embedding_dim = embedding_dim
        self.centroid = None
        self.n = 0  # Total number of bonafide samples encountered
        self.temperature = temperature  # 核心超参，建议 0.05 ~ 0.2

    def update_centroid(self, bonafide_embeddings):
        # [保持不变] 你的原始累积平均逻辑
        s = bonafide_embeddings.shape[0]
        if s == 0:
            return

        Ei = bonafide_embeddings.mean(dim=0).detach()

        if self.centroid is None:
            self.centroid = Ei
            self.n = s
        else:
            self.centroid = ((self.n * self.centroid.detach()) + (s * Ei)) / (self.n + s)
            self.n += s

    def one_class_loss(self, bonafide_embeddings, fake_embeddings):
        # [核心数学公式]: LSE (Log-Sum-Exp)
        # Loss = tau * log( mean( exp( dist / tau ) ) )
        # 这种写法数值更稳定 (利用 torch.logsumexp 避免溢出)
        # log(mean(exp(x))) = log(sum(exp(x))/N) = log(sum(exp(x))) - log(N)
        if self.centroid is None:
             return ValueError("Centroid has not been initialized with bonafide samples.")
        # Normalize
        centroid_norm = F.normalize(self.centroid.detach(), p=2, dim=0)
        Mb = bonafide_embeddings.shape[0]
        
        bonafide_norm = F.normalize(bonafide_embeddings, p=2, dim=1)
        # Sim: (B,)
        bonafide_sim = torch.matmul(bonafide_norm, centroid_norm)
        
        # Distance = 1 - Similarity
        dists = 1.0 - bonafide_sim 
        
        # 1. Scale by temperature
        scaled_dists = dists / self.temperature
        
        # 2. LogSumExp
        lse_bonafide = torch.logsumexp(scaled_dists, dim=0)
        
        # 3. Normalize by Batch Size and multiply tau
        # loss = tau * (lse - log(N))
        loss_bonafide = self.temperature * (lse_bonafide - torch.log(torch.tensor(Mb, dtype=torch.float, device=dists.device)))
        
        return loss_bonafide

    def forward(self, embeddings, labels=None, stage="train"):
        bonafide_embeddings = embeddings[labels == 1]
        fake_embeddings = embeddings[labels == 0]
        self.update_centroid(bonafide_embeddings)
        loss = self.one_class_loss(bonafide_embeddings, fake_embeddings)
        return loss