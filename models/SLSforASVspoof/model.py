# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

# %% [markdown]
# from org_codes.model import SSLModel


# %%
from transformers import AutoModelForPreTraining


import math

class DynamicSliceLinear(nn.Module):
    """动态线性层，根据输入长度使用大矩阵的对应部分"""
    
    def __init__(self, out_features, max_in_features, reference_lengths=None):
        """
        初始化可分片的动态线性层
        
        参数:
            out_features: 输出特征数量
            max_in_features: 支持的最大输入特征数量
            reference_lengths: 参考长度字典，格式为{audio_length: feature_dim}
        """
        super(DynamicSliceLinear, self).__init__()
        
        self.out_features = out_features
        self.max_in_features = max_in_features
        self.reference_lengths = reference_lengths or {48000: 16709, 64000: 22847}
        
        # 创建完整权重矩阵和偏置
        self.weight = nn.Parameter(torch.Tensor(out_features, max_in_features))
        self.bias = nn.Parameter(torch.Tensor(out_features))
        
        # 初始化参数
        self.reset_parameters()
        
    def reset_parameters(self):
        """初始化权重和偏置"""
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, x):
        """
        前向传播，根据输入尺寸动态选择权重矩阵的子集
        
        参数:
            x: 输入张量 (batch_size, in_features)
        
        返回:
            output: 线性变换后的张量 (batch_size, out_features)
        """
        in_features = x.size(1)
        
        if in_features > self.max_in_features:
            raise ValueError(f"输入特征维度 {in_features} 超过了预设的最大维度 {self.max_in_features}")
        
        # 选择权重矩阵的前in_features列
        selected_weight = self.weight[:, :in_features]
        
        # 执行线性变换
        output = F.linear(x, selected_weight, self.bias)
        return output
    
    def extra_repr(self):
        """返回额外的表示信息"""
        return f'out_features={self.out_features}, max_in_features={self.max_in_features}'
    
    

# %%
class XLS_R_SLS(nn.Module):
    
    
    def __init__(self, pretrained_path="/home/zyz/data/wav2vec2-xls-r-300m", audio_length=48000, **kwargs):
        super(XLS_R_SLS, self).__init__()

    
        # model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([pretrained_path])
        # self.ssl_model = model[0]
        self.ssl_model = AutoModelForPreTraining.from_pretrained(pretrained_path)
        
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.fc0 = nn.Linear(1024, 1)
        self.sig = nn.Sigmoid()
        # if audio_length == 48000:
        #     self.fc1 = nn.Linear(16709, 1024)
        # elif audio_length == 64000:
        #     self.fc1 = nn.Linear(22847, 1024)
        # else:
        #     raise ValueError("audio_length should be 48000 or 64000, but got {}".format(audio_length))
        self.fc1 = DynamicSliceLinear(1024, 22847) ## 最大支持 64000 长度的音频输入
        
        
        self.fc3 = nn.Linear(1024,2)
        self.logsoftmax = nn.LogSoftmax(dim=1)


    def extract_feat(self, input_data: torch.Tensor):
        
        res = self.ssl_model(input_data, output_hidden_states=True)
        hidden_states = res.hidden_states
        final_hidden_state = hidden_states[-1]
        return hidden_states[1:], final_hidden_state
    
    def getAttenF(self, layerResult):
        poollayerResult = []
        fullf = []
        for layer in layerResult:

            layery = layer.transpose(1, 2) # (B, T, C) -> (B, C, T)
            layery = F.adaptive_avg_pool1d(layery, 1) #(b,1024,1)
            layery = layery.transpose(1, 2) # (b,1,1024)
            poollayerResult.append(layery)

            fullf.append(layer[:, None, :, :]) # (B, 1, T, C)

        layery = torch.cat(poollayerResult, dim=1)
        fullfeature = torch.cat(fullf, dim=1)
        return layery, fullfeature


    def forward(self, x, return_dict=False):
        hidden_states, final_hidden_state = self.extract_feat(x)
        
        y0, fullfeature = self.getAttenF(hidden_states)
        
        y0 = self.fc0(y0)
        y0 = self.sig(y0)
        y0 = y0.view(y0.shape[0], y0.shape[1], y0.shape[2], -1) # (B, 24, 1, 1)
        # print("weight shape", y0.shape) 
        
        fullfeature = fullfeature * y0
        fullfeature = torch.sum(fullfeature, 1)
        fullfeature = fullfeature.unsqueeze(dim=1) # (B, 1, T, C)
        
        # print(y0.shape, fullfeature.shape)
        
        x = self.first_bn(fullfeature)
        x = self.selu(x)
        x = F.max_pool2d(x, (3, 3))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        feat = self.selu(x)
        
        
        logit = self.logsoftmax(self.selu(self.fc3(feat)))
        # logit = self.fc3(feat)
        if return_dict:
            return {'logit': logit, 'latent_feat': feat}
        
        return logit


# %%

# %%
# model = XLS_R_SLS()

# %%
# import torch
# x = torch.randn(3, 16000*3)
# model(x) # (3, 2)

# %%
