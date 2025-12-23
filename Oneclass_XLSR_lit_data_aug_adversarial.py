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
from myutils.zyz.aocloss import AOCloss, AOCloss_plus

class ALDA_OneClass_Adversarial_Lit(DeepfakeAudioClassification):
    def __init__(self, cfg=None, args=None, proj=False, **kwargs):
        super().__init__()
        
        self.cfg = cfg
        self.args = args
        
        # 1. 骨干网络
        self.model = XLSR_Teacher(proj=True)
        self.embed_dim = 256 
        
        # 3. 初始化Loss和优化器
        self.configure_loss_fn()
        self.configure_normalizer()
        self.register_buffer("centroid", torch.zeros(self.embed_dim))
    def generate_pgd_attack(self, audio, epsilon=0.01, alpha=0.002, num_steps=7):
        """
        生成 PGD 对抗样本 (修复版)
        """
        # 1. 复制并开启梯度
        # 注意：这里 detach() 很重要，防止梯度传回原来的 audio
        delta = torch.zeros_like(audio).uniform_(-epsilon, epsilon)
        delta.requires_grad = True
        
        # 暂时将模型设为评估模式，因为我们只想要对 Input 的梯度，不更新 BatchNorm
        # 但如果是训练对抗样本，保持 train 模式也可以，视具体策略而定
        # self.model.eval() 
        
        # 2. 迭代攻击
        for _ in range(num_steps):
            # 每次反向传播前，必须清零梯度
            # 注意：我们要清零的是 delta 的梯度，不是模型的
            if delta.grad is not None:
                delta.grad.zero_()
                
            # 前向传播
            noisy_audio = audio + delta
            
            # 计算特征和距离
            res = self.model(noisy_audio)
            feat = F.normalize(res["final_feat"], p=2, dim=1)
            centroid = F.normalize(self.loss_fn.centroid.to(feat.device), p=2, dim=0)
            
            # Loss: 我们希望 Maximize Distance (Minimize Similarity)
            # PGD 是梯度上升 (Gradient Ascent) 来增加 Loss
            # 或者梯度下降 (Gradient Descent) 来减小 Similarity
            loss = torch.matmul(feat, centroid).mean()
            
            # 反向传播，计算 d(Sim)/d(delta)
            loss.backward()
            
            with torch.no_grad():
                # --- 核心修复 ---
                # 使用 .data 进行原地更新，不破坏计算图结构
                delta_grad = delta.grad.detach()
                
                # 因为我们要让 Similarity 变小 (远离中心)，我们要沿着梯度反方向走
                # Sim 越小越好 -> x = x - alpha * sign(grad_sim)
                delta.data = delta.data - alpha * delta_grad.sign()
                
                # Projection (截断到 epsilon 球内)
                delta.data = torch.clamp(delta.data, min=-epsilon, max=epsilon)
                
                # 这一步其实不需要了，因为上面判断了 if delta.grad is not None
                # delta.grad.zero_() 
        
        # 恢复模型状态 (如果有切换的话)
        # self.model.train()
        
        return (audio + delta).detach()
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
        embeddings = F.normalize(batch_res["final_feat"], p=2, dim=1)
        
        # ---------------- CRITICAL LOGIC ----------------
        # 你的需求：训练集全是 label=1 的真实音频。
        # 你的AOCloss逻辑：labels==0 是 Bonafide (用于更新centroid)。
        # 因此，在训练计算Loss时，我们需要强制欺骗Loss函数，告诉它这些都是 class 0
        
        if self.training:
            # 训练阶段：强制所有标签为 0 (Bonafide)，以便更新 centroid 并计算距离损失
            # 这样 loss = 1 - sim(embedding, centroid)
            virtual_labels = torch.ones(embeddings.shape[0], device=embeddings.device, dtype=torch.long)
        else:
            input_label = batch["label"].type(torch.long)
            virtual_labels = input_label 
            
        loss = self.loss_fn(embeddings, virtual_labels,stage)
        # self.centroid = self.loss_fn.centroid
        return loss

    def _shared_pred(self, batch, batch_idx, stage='train', **kwargs):
        audio = batch["audio"]
        if len(audio.shape) == 3:
            audio = audio[:, 0, :]

        # 1. 获取模型输出
        res = self.model(audio)
        # 获取 Embedding (B, 1024)
        embedding = res["final_feat"]
            
        # 将增强后的音频送入同一个模型 (共享权重)    
        
        
        # 2. 计算 Logit (基于与中心点的相似度)
        if self.loss_fn.centroid == None:
            self.loss_fn.centroid = self.centroid
        centroid = self.loss_fn.centroid
        
        adv_audio = self.generate_pgd_attack(audio, epsilon=0.01) # Epsilon 可以设大一点
        res_adv = self.model(adv_audio)
        feat_noisy = res_adv["final_feat"]   

        if centroid is None:
            # 训练刚开始第一步，中心还没初始化，返回随机值防止报错
            embedding_norm = F.normalize(embedding, p=2, dim=1)
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
            "feat_noisy": feat_noisy,
        }
    def on_train_epoch_end(self):
        self.centroid = self.loss_fn.centroid
    def training_step(self, batch, batch_idx,stage="train"):
        batch_res = self._shared_eval_step(batch, batch_idx, stage="train")
        
        # 1. Anchor (Clean) -> Detach
        clean_feat = batch_res["final_feat"]
        clean_norm = F.normalize(clean_feat, p=2, dim=1).detach()
        
        # 2. View (Noisy)
        # 将增强后的音频送入同一个模型 (共享权重)
        feat_noisy = batch_res["feat_noisy"]
        noisy_norm = F.normalize(feat_noisy, p=2, dim=1)
        
        # 3. Consistency Loss
        similarity = (clean_norm * noisy_norm).sum(dim=1)
        loss_consistency = (1 - similarity).mean()
        
        # 4. Combine
        loss_cluster = batch_res["loss"]
        loss = 5.0 * loss_cluster +1.0 * loss_consistency 
        # loss = torch.relu(10*loss_cluster - loss_consistency + 0.02)       
        self.log("train/loss_cluster", loss_cluster)
        self.log("train/loss_consistency", loss_consistency)
        
        batch_res["loss"] = loss
        if not isinstance(loss, dict):
            loss = {"loss": loss,
                    "loss_cluster": loss_cluster,
                    "loss_consistency": loss_consistency}

        suffix = ""
        self.log_dict(
            {f"{stage}-{key}{suffix}": loss[key] for key in loss},
            # on_step=True if stage=='train' else False,
            on_step=False,
            on_epoch=True,
            logger=True,
            prog_bar=True,
            add_dataloader_idx=False,
            batch_size=batch["label"].shape[0],
        )
        return batch_res
    