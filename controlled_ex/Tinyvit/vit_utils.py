import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from .tiny_vit import tiny_vit_21m_224 # 假设这是你的模型文件路径
# from .transforms import SpecAugmentBatchTransform # 假设的增强路径

class MultiStageLayerWeighting(nn.Module):
    """
    功能：学习每一层的权重，将多层特征融合。
    保持不变。
    """
    def __init__(self, input_dim=576, num_stages=4):
        super().__init__()
        self.fc_weight = nn.Linear(input_dim, 1) 
        self.softmax = nn.Softmax(dim=1) 

    def forward(self, hidden_states_list):
        # hidden_states_list: List of (B, 49, 576)
        full_features = torch.stack(hidden_states_list, dim=1) # (B, 4, 49, 576)
        layer_reps = torch.mean(full_features, dim=2) # (B, 4, 576)
        raw_scores = self.fc_weight(layer_reps) 
        layer_weights = self.softmax(raw_scores).unsqueeze(-1) # (B, 4, 1, 1)
        fused_feature = torch.sum(full_features * layer_weights, dim=1) # (B, 49, 576)
        return fused_feature, layer_weights.squeeze()

class PatchAttentionRefinement(nn.Module):
    """
    功能：Patch级注意力精炼。
    保持不变。
    """
    def __init__(self, input_dim=576, attention_channels=128):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(input_dim, attention_channels, kernel_size=1),
            nn.Tanh(), 
            nn.Conv1d(attention_channels, 1, kernel_size=1)
        )

    def forward(self, x):
        # x: (B, 49, 576)
        x_in = x.transpose(1, 2) # (B, 576, 49)
        attn_scores = self.attention(x_in) # (B, 1, 49)
        attn_weights = F.softmax(attn_scores, dim=2) 
        refined_feat = x_in * attn_weights
        return refined_feat.transpose(1, 2), attn_weights