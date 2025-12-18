import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForPreTraining

class ALDA_Teacher_Strict(nn.Module):
    def __init__(self):
        super(ALDA_Teacher_Strict, self).__init__()
        
        # 1. 加载 XLS-R (冻结)
        self.ssl_model = AutoModelForPreTraining.from_pretrained("/home/zyz/data/wav2vec2-xls-r-300m")
        # self.ssl_model.eval()
        # for param in self.ssl_model.parameters():
        #     param.requires_grad = False
            
        # 2. 严格复刻 XLS_R_SLS 的层权重部分
        # 原代码：self.fc0 = nn.Linear(1024, 1) + Sigmoid
        self.layer_fc = nn.Linear(1024, 1)
        self.sig = nn.Sigmoid()
        
        # 3. 严格复刻 ASP 的注意力计算部分 (但不做 Pooling)
        # 原代码使用 bottleneck: input -> 128 -> 1
        self.attn_conv1 = nn.Conv1d(1024, 128, kernel_size=1)
        self.attn_tanh = nn.Tanh()
        self.attn_conv2 = nn.Conv1d(128, 1, kernel_size=1)

    def get_layer_weights(self, hidden_states):
        """
        复刻 XLS_R_SLS 的 getAttenF 逻辑
        """
        # hidden_states: tuple of (B, T, 1024)
        
        # 步骤 A: 对每一层做全局平均池化 (B, T, 1024) -> (B, 1024)
        # 对应原代码：adaptive_avg_pool1d
        pooled_layers = []
        full_layers = []
        
        for layer in hidden_states:
            # 原代码逻辑：(B, T, C) -> (B, C, T) -> pool -> (B, C, 1) -> (B, 1, C)
            # 简化写法但数学等价：
            pooled = layer.mean(dim=1) # (B, 1024)
            pooled_layers.append(pooled)
            full_layers.append(layer) # (B, T, 1024)
            
        # 堆叠: (B, 24, 1024)
        layer_stack = torch.stack(pooled_layers, dim=1)
        full_stack = torch.stack(full_layers, dim=1) # (B, 24, T, 1024)
        
        # 步骤 B: 计算权重 (Linear + Sigmoid)
        # (B, 24, 1024) -> (B, 24, 1)
        weights = self.layer_fc(layer_stack)
        weights = self.sig(weights) # 使用 Sigmoid 保持原汁原味
        
        # 步骤 C: 加权求和
        # (B, 24, T, 1024) * (B, 24, 1, 1) -> Sum -> (B, T, 1024)
        fused_feat = torch.sum(full_stack * weights.unsqueeze(2), dim=1)
        
        return fused_feat, weights.squeeze()

    def apply_temporal_attention(self, x):
        """
        复刻 AttentiveStatisticsPooling 的计算逻辑，但改为输出序列
        """
        # x: (B, T, 1024) -> 转置为 (B, 1024, T) 以适配 Conv1d
        x_in = x.transpose(1, 2)
        
        # 步骤 A: 计算 Attention Score (公式与 ASP 一致)
        w = self.attn_conv1(x_in)
        w = self.attn_tanh(w)
        w = self.attn_conv2(w) # (B, 1, T)
        
        # 步骤 B: Softmax 归一化 (ASP 也是用的 Softmax)
        attn_weights = F.softmax(w, dim=2)
        
        # 步骤 C: 应用权重 (关键修改：不做 sum pooling，而是 element-wise 乘法)
        # 我们需要保留 (B, T, D) 给 Student 学习
        refined_x = x_in * attn_weights
        
        return refined_x.transpose(1, 2), attn_weights

    def forward(self, x):
        # 1. SSL Forward
        # with torch.no_grad():
        res = self.ssl_model(x, output_hidden_states=True)
        # 取后24层 (跳过 embedding 层)
        hidden_states = res.hidden_states[1:] 
            
        # 2. 层级加权
        feat, layer_w = self.get_layer_weights(hidden_states)
        
        # 3. 时序加权
        final_target, time_w = self.apply_temporal_attention(feat)
        
        return {
            "final_feat": final_target, # (B, T, 1024) -> 给 Student 的标签
            "layer_weights": layer_w,
            "time_weights": time_w
        }