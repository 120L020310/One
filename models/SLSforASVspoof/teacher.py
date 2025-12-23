import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForPreTraining

from models.MultiView.utils import Hubert_ASR

class DynamicLayerWeighting(nn.Module):
    """
    基于用户提供的 XLS_R_SLS 代码改编。
    功能：学习每一层的权重，将多层特征融合为单一特征图。
    """
    def __init__(self, input_dim=1024, num_layers=25): # XLS-R-300m 有 24 hidden layers + 1 input embedding
        super(DynamicLayerWeighting, self).__init__()
        # 用于计算层权重的线性层
        # 输入：(B, Layers, Dim) -> Pool -> (B, Layers, Dim) -> Linear -> (B, Layers, 1)
        self.fc_weight = nn.Linear(input_dim, 1) 
        self.softmax = nn.Softmax(dim=1) # 保证所有层权重之和为1

    def forward(self, hidden_states):
        """
        hidden_states: tuple of tensors, len = 25. Each tensor: (B, T, D)
        """
        # 1. 堆叠所有层: (B, T, D) * L -> (B, L, T, D)
        # 注意：为了节省显存，可以先 detach 不需要梯度的部分（如果 XLS-R 是冻结的）
        full_features = torch.stack(hidden_states, dim=1) # (B, L, T, D)
        
        # 2. 计算层权重 (Layer Attention)
        # 策略：对 T 维度做全局平均池化，得到每一层的整体表示
        # (B, L, T, D) -> (B, L, D)
        layer_reps = torch.mean(full_features, dim=2) 
        
        # 通过线性层计算分数: (B, L, D) -> (B, L, 1)
        raw_scores = self.fc_weight(layer_reps) 
        
        # 归一化权重: (B, L, 1, 1) 用于广播
        layer_weights = self.softmax(raw_scores).unsqueeze(-1)
        
        # 3. 加权求和 (Weighted Sum)
        # (B, L, T, D) * (B, L, 1, 1) -> sum dim 1 -> (B, T, D)
        fused_feature = torch.sum(full_features * layer_weights, dim=1)
        
        return fused_feature, layer_weights.squeeze()

class TemporalAttentionRefinement(nn.Module):
    """
    基于用户提供的 AttentiveStatisticsPooling 改编。
    功能：不进行Pooling，而是利用 Attention Map 对特征进行时序上的'高亮'。
    """
    def __init__(self, input_dim=1024, attention_channels=128):
        super(TemporalAttentionRefinement, self).__init__()
        # 对应代码2中的 bottleneck 结构
        self.attention = nn.Sequential(
            nn.Conv1d(input_dim, attention_channels, kernel_size=1),
            nn.Tanh(), 
            nn.Conv1d(attention_channels, 1, kernel_size=1)
        )

    def forward(self, x):
        """
        x: (B, T, D) - 来自 LayerWeighting 的输出
        """
        # 调整维度适配 Conv1d: (B, T, D) -> (B, D, T)
        x_in = x.transpose(1, 2)
        
        # 1. 计算 Attention Scores
        attn_scores = self.attention(x_in)  # [B, 1, T]
        
        # 2. 计算归一化权重 alpha_t
        attn_weights = F.softmax(attn_scores, dim=2) # [B, 1, T]
        
        # 3. 时序加权 (Refinement)
        # 这里我们不做 Pooling，而是将权重乘回原特征
        # 这相当于告诉后续的 Student：“这一帧很重要，你要重点模仿；那一帧是静音，可以忽略”
        # (B, D, T) * (B, 1, T) -> (B, D, T)
        refined_feat = x_in * attn_weights
        
        # 转回 (B, T, D)
        return refined_feat.transpose(1, 2), attn_weights

class XLSR_Teacher(nn.Module):
    """
    方案一的完整 Teacher 模型封装
    结构：Frozen XLS-R -> Learnable Layer Weighting -> Learnable Temporal Attention
    """
    def __init__(self,proj=False):
        super(XLSR_Teacher, self).__init__()
        
        # 1. Backbone: XLS-R (冻结)
        self.xlsr = AutoModelForPreTraining.from_pretrained("/home/zyz/data/wav2vec2-xls-r-300m")
        input_dim = 1024 # XLS-R-300m output dim
        
        # 学习每一层的重要性
        self.layer_weighting = DynamicLayerWeighting(input_dim=input_dim, num_layers=25)
        
        # 学习每一帧的重要性
        self.temporal_refiner = TemporalAttentionRefinement(input_dim=input_dim, attention_channels=128)
        if proj:
            self.projector = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.LayerNorm(input_dim),
                nn.ReLU(inplace=True),
                nn.Linear(input_dim, 256)  # 降维通常有助于聚类，设为 256 或 128
            )
        self.proj=proj

    def forward(self, x):
        if x.dim()==3:
            x = x.squeeze(1)
        outputs = self.xlsr(x, output_hidden_states=True)
        hidden_states = outputs.hidden_states[1:] 
        
        # 2. 动态层融合 (Gradient 流经此处)
        fused_feat, layer_w = self.layer_weighting(hidden_states)
        
        # 3. 时序注意力精炼 (Gradient 流经此处)
        refined_feat, time_w = self.temporal_refiner(fused_feat)
        representation = refined_feat.mean(dim=1)
        if self.proj:
            representation = self.projector(representation)
        # 也可以返回 layer_w 和 time_w 用于可视化分析
        return {
            'final_feat': representation,  # Shape: (B, D)
            'hidden_states': hidden_states,
            'layer_weights': layer_w,      # Shape: (B, 25)
            'time_weights': time_w         # Shape: (B, 1, T)
        }

class Hubert_Student(nn.Module):
    """
    方案一的完整 Teacher 模型封装
    结构：Frozen XLS-R -> Learnable Layer Weighting -> Learnable Temporal Attention
    """
    def __init__(self, proj=False):
        super(Hubert_Student, self).__init__()
        
        # 1. Backbone: XLS-R (冻结)
        self.hubert = Hubert_ASR()
        input_dim = 1024 # XLS-R-300m output dim
        
        # 学习每一层的重要性
        self.layer_weighting = DynamicLayerWeighting(input_dim=input_dim, num_layers=24)
        
        # 学习每一帧的重要性
        self.temporal_refiner = TemporalAttentionRefinement(input_dim=input_dim, attention_channels=128)
        if proj:
            self.projector = nn.Sequential(
                nn.Linear(input_dim, input_dim),
                nn.LayerNorm(input_dim),
                nn.ReLU(inplace=True),
                nn.Linear(input_dim, input_dim)  # 降维通常有助于聚类，设为 256 或 128
            )
        self.proj=proj

    def forward(self, x):
        if x.dim()==3:
            x = x.squeeze(1)
        final_feat, hidden_states = self.hubert(x)
        
        # 2. 动态层融合 (Gradient 流经此处)
        fused_feat, layer_w = self.layer_weighting(hidden_states)
        
        # 3. 时序注意力精炼 (Gradient 流经此处)
        refined_feat, time_w = self.temporal_refiner(fused_feat)
        representation = refined_feat.mean(dim=1)
        if self.proj:
            representation = self.projector(representation)
        return {
            'final_feat': representation,  # Shape: (B, D)
            'hidden_states': hidden_states,
            'layer_weights': layer_w,      # Shape: (B, 25)
            'time_weights': time_w         # Shape: (B, 1, T)
        }