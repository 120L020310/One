import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from controlled_ex.Tinyvit.model import AddGaussianNoise
from controlled_ex.Tinyvit.tiny_vit import tiny_vit_21m_224
from controlled_ex.Tinyvit.vit_utils import MultiStageLayerWeighting, PatchAttentionRefinement
from myutils.torchaudio.transforms._SpecAug import SpecAugmentBatchTransform

class TinyVit_Teacher(nn.Module):
    def __init__(self, verbose=0, pretrained=True):
        super().__init__()
        
        # 1. Backbone
        self.model = tiny_vit_21m_224(pretrained=pretrained)
        self.spectrogram = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=187)
        self.verbose = verbose
        # self.spec_aug = SpecAugmentBatchTransform.from_policy("ss") 
        self.spec_aug = None

        # -----------------------------------------------------------
        # 新增：特征对齐层 (Projectors & Poolers)
        # 目标维度: (49, 576)
        
        target_dim = 576
        
        # Layer 0: 192 -> 576
        self.proj0 = nn.Linear(192, target_dim)
        # Layer 1: 384 -> 576
        self.proj1 = nn.Linear(384, target_dim)
        # Layer 2 & 3: 576 -> 576 (Identity, 不需要Linear)
        
        # 空间对齐: 强制下采样到 7x7 (=49 patches)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((7, 7))

        # 2. 附加模块：层级融合
        self.layer_weighting = MultiStageLayerWeighting(input_dim=target_dim, num_stages=4)
        
        # 3. 附加模块：Patch级精炼
        self.patch_refiner = PatchAttentionRefinement(input_dim=target_dim, attention_channels=128)

    def preprocess(self, x, stage="test"):
        x = self.spectrogram(x.unsqueeze(1))
        if stage == "train":
            x = self.noise_adder(x)
        
        x = F.interpolate(x, size=(224, 224), mode="bilinear")
        x = torch.log(x + 1e-7)
        
        x = (x - torch.mean(x, dim=(1, 2, 3), keepdim=True)) / (
            torch.std(x, dim=(1, 2, 3), keepdim=True) + 1e-9
        )
        if stage == "train" and self.spec_aug is not None:
             x = self.spec_aug.batch_apply(x)
        return x

    def align_feature(self, x, projector=None):
        """
        通用对齐函数：
        1. 恢复 2D 空间结构 (H, W)
        2. 池化到 7x7
        3. 展平
        4. 投影通道
        """
        B, N, C = x.shape
        # 计算 H, W (假设是正方形)
        H = int(math.sqrt(N))
        W = H
        
        # 1. 转换回图像格式以便池化: (B, N, C) -> (B, C, N) -> (B, C, H, W)
        x = x.transpose(1, 2).view(B, C, H, W)
        
        # 2. 空间对齐 -> (B, C, 7, 7)
        # 如果当前尺寸已经是 7x7 (N=49)，adaptive_pool 相当于没做操作，很安全
        x = self.adaptive_pool(x) 
        
        # 3. 展平回序列: (B, C, 49) -> (B, 49, C)
        x = x.flatten(2).transpose(1, 2)
        
        # 4. 通道对齐
        if projector is not None:
            x = projector(x)
            
        return x
    
    def forward(self, x, stage="test"):
        # 1. 预处理
        x = self.preprocess(x, stage)
        x = torch.cat([x, x, x], dim=1)
        # 2. Patch Embed
        x = self.model.patch_embed(x) 
        aligned_features = []
        # --- Layer 0 ---
        x = self.model.layers[0](x) 
        # Output: [B, 784, 192] -> Align to [B, 49, 576]
        feat0 = self.align_feature(x, self.proj0)
        aligned_features.append(feat0)
        
        # --- Layer 1 ---
        x = self.model.layers[1](x)
        # Output: [B, 196, 384] -> Align to [B, 49, 576]
        feat1 = self.align_feature(x, self.proj1)
        aligned_features.append(feat1)
        
        # --- Layer 2 ---
        x = self.model.layers[2](x)
        # Output: [B, 49, 576] -> Align (Spatial ok, Channel ok)
        # 虽然尺寸对了，但为了代码一致性调用 align_feature (projector=None)，
        # 这样可以确保它经过了 adaptive_pool (虽然是 7x7->7x7) 
        feat2 = self.align_feature(x, projector=None)
        aligned_features.append(feat2)
        
        # --- Layer 3 ---
        x = self.model.layers[3](x)
        # Output: [B, 49, 576] -> Align
        feat3 = self.align_feature(x, projector=None)
        aligned_features.append(feat3)

        # 4. 动态层融合 (Layer Attention)
        # Input: List of [B, 49, 576]
        fused_feat, layer_w = self.layer_weighting(aligned_features)
        
        # 5. Patch 级注意力精炼 (Patch Attention)
        # Input: [B, 49, 576]
        final_feat, patch_w = self.patch_refiner(fused_feat)
        
        # 6. 返回结果
        return {
            'final_feat': final_feat,   # Shape: (B, 49, 576)
            'layer_weights': layer_w,   # Shape: (B, 4)
            'time_weights': patch_w     # Shape: (B, 1, 49)
        }