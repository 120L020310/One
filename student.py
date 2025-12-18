import torch
import torch.nn as nn

class SimpleStudent(nn.Module):
    """
    轻量级 Student 网络
    结构：多层 1D-CNN，逐步下采样以匹配 XLS-R 的时间分辨率。
    目标：输入 Raw Audio，输出 (B, T, 1024)
    """
    def __init__(self, output_dim=1024):
        super(SimpleStudent, self).__init__()
        
        # XLS-R 的下采样倍率约为 320 (16000Hz -> 50Hz)
        # 我们使用 6 层卷积，Strides 分别为: 5, 2, 2, 2, 2, 2 -> 5*2^5 = 160 (还需要再除以2或者调整padding，近似匹配)
        # 实际操作中，为了对齐方便，通常会在最后加一个 Interpolation 或者 AdaptivePool
        
        self.encoder = nn.Sequential(
            # Layer 1: Stride 5
            nn.Conv1d(1, 128, kernel_size=10, stride=5, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            # Layer 2-6: Stride 2 (总共下采样 5*2*2*2*2*2 = 160)
            self._make_layer(128, 256),
            self._make_layer(256, 256),
            self._make_layer(256, 512),
            self._make_layer(512, 512),
            self._make_layer(512, 512),
        )
        
        # 最后的投影层，调整到 1024 维
        self.projection = nn.Conv1d(512, output_dim, kernel_size=3, padding=1)
        
    def _make_layer(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(out_c),
            nn.ReLU()
        )

    def forward(self, x):
        # x: (B, T_raw) -> (B, 1, T_raw)
        if x.dim() == 2:
            x = x.unsqueeze(1)
            
        feat = self.encoder(x) # (B, 512, T_downsampled)
        feat = self.projection(feat) # (B, 1024, T_downsampled)
        
        # 转置为 (B, T, D) 以匹配 Teacher
        return feat.transpose(1, 2)
    
class ALDALoss(nn.Module):
    def __init__(self):
        super(ALDALoss, self).__init__()
        # 余弦相似度损失
        self.cosine_loss = nn.CosineSimilarity(dim=2) 

    def forward(self, teacher_out, student_out):
        """
        teacher_out: dict from ALDA_Teacher
        student_out: (B, T_s, D)
        """
        t_feat = teacher_out['final_feat'] # (B, T_t, D)
        time_weights = teacher_out['time_weights'] # (B, 1, T_t)
        
        # 1. 维度对齐 (Alignment)
        # 由于卷积计算可能有Padding差异，T_t 和 T_s 可能差几个像素
        # 我们截取最小长度
        min_len = min(t_feat.shape[1], student_out.shape[1])
        t_feat = t_feat[:, :min_len, :]
        student_out = student_out[:, :min_len, :]
        time_weights = time_weights[:, :, :min_len].squeeze(1) # (B, T)
        
        # 2. 计算余弦相似度 (值域 [-1, 1], 1表示完全重合)
        # 我们希望 sim 接近 1，所以 loss = 1 - sim
        sim = self.cosine_loss(t_feat, student_out) # (B, T)
        loss_map = 1 - sim 
        
        # 3. 加权 Loss (Attention-guided Loss)
        # 让 Student 重点学习权重高的帧
        weighted_loss = loss_map * time_weights 
        
        return weighted_loss.mean()