import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.func import functional_call
import torchaudio
from Data_Aug import DigitalArtifactAugmentor, RobustDataAugmentor, SafeRawAugmentor, TransformerHardAugmentor
from models.SLSforASVspoof.teacher import XLSR_Teacher
from myutils.torch.deepfake_detection.audio import DeepfakeAudioClassification
from myutils.zyz.aocloss import AOCloss, AOCloss_plus, AOCloss_plus_soft
from myutils.zyz.aocloss_tight import AOCloss_AugImmunity

class ALDA_OneClass_AugImmunity_Lit(DeepfakeAudioClassification):
    def __init__(self, cfg=None, args=None, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.model = XLSR_Teacher()
        self.embed_dim = 128 
        
        # 在主类中注册 buffer，用于模型保存和加载
        self.register_buffer("centroid", torch.zeros(self.embed_dim))
        self.augmentor = SafeRawAugmentor(noise_intensity=0.12, mask_ratio=0.1)
        self.configure_loss_fn()

    def configure_optimizers(self):
        # 保持和你的一致，通常Metric Learning的学习率需要调得比CrossEntropy小一点
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-6, weight_decay=1e-4)
        return optimizer
    
    def configure_loss_fn(self):
        # 实例化自定义 Loss
        self.loss_fn = AOCloss(embedding_dim=self.embed_dim)
        
        # 如果是加载模型，初始化时把 buffer 的值给 loss 内部
        if torch.any(self.centroid != 0):
            self.loss_fn.centroid = self.centroid.clone()

    def _shared_pred(self, batch, batch_idx, stage='train', **kwargs):
        audio = batch["audio"]
        label = batch["label"]
        if len(audio.shape) == 3:
            audio = audio[:, 0, :]

        # 1. 获取模型输出
        res = self.model(audio)
        # 获取 Embedding (B, 1024)
        feat_clean = res["final_feat"]
        # 带噪的前向传播
        audio_noisy_input = self.augmentor(audio)
        
        # 将增强后的音频送入同一个模型 (共享权重)
        res_noisy = self.model(audio_noisy_input)
        feat_noisy = res_noisy["final_feat"]    
        
        # noisy：永远不更新统计量（避免污染）
        # 2. 计算 Logit (基于与中心点的相似度)
        if self.loss_fn.centroid == None:
            self.loss_fn.centroid = self.centroid
        centroid = self.loss_fn.centroid
        


        if centroid is None:
            raise KeyError
        else:
            # 归一化
            embedding_norm = F.normalize(feat_clean, p=2, dim=1)
            centroid_norm = F.normalize(centroid.to(feat_clean.device), p=2, dim=0)
            
            # 计算余弦相似度 [Batch]
            # similarity range: [-1, 1]
            similarity = torch.matmul(embedding_norm, centroid_norm)

            logit = similarity
        batch_pred = (logit > 0.95).int()

        return {
            "final_feat": feat_clean,
            "feat_noisy": feat_noisy, 
            "logit": logit,
            "pred": batch_pred
        }
    
    def calcuate_loss(self, batch_res, batch, stage):
        embeddings = F.normalize(batch_res["final_feat"], p=2, dim=1)     
        if self.training:
            # 训练阶段：强制所有标签为 0 (Bonafide)，以便更新 centroid 并计算距离损失
            # 这样 loss = 1 - sim(embedding, centroid)
            virtual_labels = torch.ones(embeddings.shape[0], device=embeddings.device, dtype=torch.long)
        else:
            input_label = batch["label"].type(torch.long)
            virtual_labels = input_label 
            
        loss_cluster = self.loss_fn(embeddings, virtual_labels,stage)
        # 3. Consistency Loss
        clean_norm = F.normalize(batch_res["final_feat"], p=2, dim=1)
        noisy_norm = F.normalize(batch_res["feat_noisy"], p=2, dim=1).detach()
        similarity = (clean_norm * noisy_norm).sum(dim=1)
        loss_consistency = (1 - similarity).mean()

        loss = 5.0 * loss_cluster + 1.0 * loss_consistency 
        return loss


    def on_train_epoch_end(self):
        self.centroid = self.loss_fn.centroid
