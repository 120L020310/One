import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

# 假设这些是你原本的引用
from models.SLSforASVspoof.teacher import XLSR_Teacher
from myutils.torch.deepfake_detection.audio import DeepfakeAudioClassification
from myutils.zyz.aocloss import AOCloss, AOCloss_plus, AOCloss_plus_soft

class XLSR_OneClass_Lit(DeepfakeAudioClassification):
    def __init__(self, cfg=None, args=None, **kwargs):
        super().__init__()
        
        self.cfg = cfg
        self.args = args
        
        # 1. 骨干网络
        self.model = XLSR_Teacher()
        
        # 2. 特征维度 (XLS-R通常是1024，需根据ALDA_Teacher实际输出调整)
        self.embed_dim = 128 
        
        # 3. 初始化Loss和优化器
        self.configure_loss_fn()
        self.configure_normalizer()
        self.register_buffer("centroid", torch.zeros(self.embed_dim))

    def configure_loss_fn(self):
        # 初始化自定义的 AOC Loss
        # 注意：Loss内部维护了 centroid
        self.loss_fn = AOCloss(embedding_dim=self.embed_dim)
        
    def configure_optimizers(self):
        # 保持和你的一致，通常Metric Learning的学习率需要调得比CrossEntropy小一点
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-6, weight_decay=1e-4)
        return optimizer

    def calcuate_loss(self, batch_res, batch,stage):
        """
        计算单类损失
        """
        # 获取特征 [Batch, 1024]
        # 使用 mean pooling 将 (B, C, T) -> (B, C)
        embeddings = batch_res["final_feat"]
        embedding_norm = F.normalize(embeddings, p=2, dim=1)
        
        # ---------------- CRITICAL LOGIC ----------------
        # 你的需求：训练集全是 label=1 的真实音频。
        # 你的AOCloss逻辑：labels==0 是 Bonafide (用于更新centroid)。
        # 因此，在训练计算Loss时，我们需要强制欺骗Loss函数，告诉它这些都是 class 0
        
        if self.training:
            # 训练阶段：强制所有标签为 0 (Bonafide)，以便更新 centroid 并计算距离损失
            # 这样 loss = 1 - sim(embedding, centroid)
            virtual_labels = torch.ones(embedding_norm.shape[0], device=embedding_norm.device, dtype=torch.long)
        else:
            input_label = batch["label"].type(torch.long)
            virtual_labels = input_label 
            
        loss = self.loss_fn(embedding_norm, virtual_labels,stage)
        # self.centroid = self.loss_fn.centroid
        return loss

    def _shared_pred(self, batch, batch_idx, stage='train', **kwargs):
        audio = batch["audio"]
        labels = batch["label"]
        if len(audio.shape) == 3:
            audio = audio[:, 0, :]

        # 1. 获取模型输出
        res = self.model(audio)
        # 获取 Embedding (B, 1024)
        embedding = res["final_feat"]
        # 2. 计算 Logit (基于与中心点的相似度)
        centroid = self.loss_fn.centroid if self.loss_fn.centroid is not None else self.centroid
        
        if centroid is None:
            # 训练刚开始第一步，中心还没初始化，返回随机值防止报错
            batch_size = embedding.shape[0]
            # 返回 0 (相似度为0)
            logit = torch.zeros(batch_size, device=embedding.device)
        else:
            # 归一化
            embedding_norm = F.normalize(embedding, p=2, dim=1)
            centroid_norm = F.normalize(centroid.to(embedding.device), p=2, dim=0)
            
            # 计算余弦相似度 [Batch]
            # similarity range: [-1, 1]
            similarity = torch.matmul(embedding_norm, centroid_norm)

            logit = similarity
        batch_pred = (logit > 0.95).int() 
        
        # 这里的 output 结构保持和你原来的一致
        return {
            "final_feat": embedding,
            "logit": logit, # 用于计算 EER, AUC 等指标
            "pred": batch_pred, # 硬预测标签
            "embedding": embedding # 额外返回 embedding 用于调试或t-SNE可视化
        }
    def on_train_epoch_end(self):
        self.centroid = self.loss_fn.centroid
    # def training_step(self, batch, batch_idx):
    #     x, y = batch['audio'], batch['label']
        
    #     # === 步骤 1: 确保模型处于 train 模式 ===
    #     # 这确保了 Dropout 层是开启的，每次 Forward 的掩码都不同
    #     self.model.train()
        
    #     # === 步骤 2: 生成两个视角的特征 ===
        
    #     # 第一次 Forward (View 1)
    #     # 虽然叫 clean，但在 train 模式下它其实也包含随机 Dropout
    #     embedding = self.model(x)['final_feat'].mean(1)
    #     # 2. 计算 Logit (基于与中心点的相似度)
    #     centroid = self.loss_fn.centroid
        
    #     if centroid is None:
    #         # 训练刚开始第一步，中心还没初始化，返回随机值防止报错
    #         batch_size = embedding.shape[0]
    #         # 返回 0 (相似度为0)
    #         logit = torch.zeros(batch_size, device=embedding.device)
    #     else:
    #         # 归一化
    #         embedding_norm = F.normalize(embedding, p=2, dim=1)
    #         centroid_norm = F.normalize(centroid.to(embedding.device), p=2, dim=0)
            
    #         # 计算余弦相似度 [Batch]
    #         # similarity range: [-1, 1]
    #         similarity = torch.matmul(embedding_norm, centroid_norm)

    #         logit = similarity
    #     batch_res = {
    #         "final_feat": embedding,
    #         "logit": logit, # 用于计算 EER, AUC 等指标
    #         "embedding": embedding # 额外返回 embedding 用于调试或t-SNE可视化
    #     }
    #     loss_aoc = self.calcuate_loss(batch_res, batch,stage="train")
    #     # 第二次 Forward (View 2)
    #     # 由于 Dropout 掩码是动态生成的，feat_2 会和 feat_1 略有不同（互为增强）
    #     feat_2 = self.model(x)['final_feat'].mean(1)
        
    #     # === 步骤 3: 计算损失 ===
        
    #     # B. 一致性 Loss: 强迫 feat_1 和 feat_2 保持一致
    #     # 核心逻辑：同一个样本，即使经过不同的 Dropout，特征也不应该剧烈跳变
    #     loss_consistency = F.mse_loss(embedding, feat_2)
        
    #     # === 步骤 4: 融合 ===
    #     # lambda 系数建议从 5.0 开始尝试
    #     total_loss = loss_aoc + 5.0 * loss_consistency 
        
    #     self.log("train_loss", total_loss)
    #     self.log("consistency_loss", loss_consistency)
        
    #     return {
    #         "final_feat": embedding,
    #         "logit": logit, # 用于计算 EER, AUC 等指标
    #         "embedding": embedding, # 额外返回 embedding 用于调试或t-SNE可视化
    #         "loss":total_loss
    #     }