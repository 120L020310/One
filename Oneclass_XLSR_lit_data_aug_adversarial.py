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
        生成 PGD 对抗样本 (最终修复版：解决 validation 报错 + 梯度污染)
        """
        # --- 1. 准备工作 ---
        original_mode = self.model.training 
        self.model.eval() # 必须 eval，冻结 BN 统计量
        
        # 暂时关闭模型参数的梯度计算
        # 注意：这仅仅是不计算权重的梯度，但我们需要计算图来回传给 Input
        for param in self.model.parameters():
            param.requires_grad = False

        # --- 2. 强制开启梯度上下文 (关键修复) ---
        # 即使在 validation_step (默认 no_grad) 中调用，这里也会强制开启计算图构建
        with torch.enable_grad():
            
            # 初始化扰动
            delta = torch.zeros_like(audio).uniform_(-epsilon, epsilon)
            delta.requires_grad = True
            
            # --- 3. 迭代攻击 ---
            for _ in range(num_steps):
                if delta.grad is not None:
                    delta.grad.zero_()
                    
                # 前向传播
                noisy_audio = audio + delta
                res = self.model(noisy_audio)
                
                # 计算特征和距离
                feat = F.normalize(res["final_feat"], p=2, dim=1)
                
                # 注意：self.loss_fn.centroid 也是 Parameter，需要 detach 或者保证不求导
                centroid = F.normalize(self.loss_fn.centroid.to(feat.device), p=2, dim=0)
                
                # Loss = Cosine Similarity (目标: 最小化 Sim)
                loss = torch.matmul(feat, centroid).mean()
                
                # 反向传播 (因为有 enable_grad，这里 loss 会有 grad_fn，不会报错了)
                loss.backward()
                
                # PGD 更新 (In-place update)
                with torch.no_grad():
                    delta_grad = delta.grad.detach()
                    delta.data = delta.data - alpha * delta_grad.sign()
                    delta.data = torch.clamp(delta.data, min=-epsilon, max=epsilon)
        
        # --- 4. 收尾工作 ---
        # 恢复模型参数的梯度需求
        for param in self.model.parameters():
            param.requires_grad = True
            
        # 恢复模型原本的训练状态
        self.model.train(original_mode)
        self.model.zero_grad() 
        
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
    def validation_step(self, batch, batch_idx):
        # 1. 执行前向传播 (包含 PGD 攻击)
        # 必须开启梯度以生成对抗样本
        with torch.enable_grad():
            batch_res = self._shared_pred(batch, batch_idx, stage="val")
        
        # 2. 获取特征 [CRITICAL FIX: DETACH]
        # 必须使用 .detach()，否则计算图会一直保留在显存中导致 OOM
        clean_feat = batch_res["final_feat"].detach()
        noisy_feat = batch_res["feat_noisy"].detach()
        
        # 3. 归一化
        clean_norm = F.normalize(clean_feat, p=2, dim=1)
        noisy_norm = F.normalize(noisy_feat, p=2, dim=1)
        
        # 确保 centroid 在正确的设备上
        centroid = self.centroid.to(clean_feat.device)
        centroid = F.normalize(centroid, p=2, dim=0)

        # 4. 计算两个分数
        # Score 1: 聚类分数 (离中心越远越假)
        # Real 靠近中心 -> sim 大 -> dist 小
        dist_cluster = 1.0 - torch.matmul(clean_norm, centroid)
        
        # Score 2: 一致性分数 (被攻击后跑得越远越假)
        # Real 免疫攻击 -> sim 大 -> dist 小
        dist_consistency = 1.0 - (clean_norm * noisy_norm).sum(dim=1)
        
        # 5. 融合分数 (Anomaly Score)
        # 这是一个超参数，建议你在初期可以分别观察两个分数，或者设为可调参数
        # 这里的 2.0 意味着你更看重 "鲁棒性" 这一特征
        anomaly_score = dist_cluster + 2.0 * dist_consistency
        
        # 6. 记录 Log 用于监控 (分项记录非常有必要)
        # 使用 batch_size 参数确保 log 准确
        bs = batch['label'].shape[0]
        
        # 记录 Real 样本的分数 (越低越好)
        real_mask = (batch['label'] == 1)
        if real_mask.sum() > 0:
            self.log("val/score_cluster_real", dist_cluster[real_mask].mean(), batch_size=bs)
            self.log("val/score_consist_real", dist_consistency[real_mask].mean(), batch_size=bs)
            self.log("val/total_score_real", anomaly_score[real_mask].mean(), batch_size=bs)

        # 记录 Fake 样本的分数 (越高越好)
        fake_mask = (batch['label'] == 0)
        if fake_mask.sum() > 0:
            self.log("val/score_cluster_fake", dist_cluster[fake_mask].mean(), batch_size=bs)
            self.log("val/score_consist_fake", dist_consistency[fake_mask].mean(), batch_size=bs)
            self.log("val/total_score_fake", anomaly_score[fake_mask].mean(), batch_size=bs)

        # 7. 返回结果
        # 返回 detach 后的 tensor 或 cpu tensor，方便后续 concat 计算 AUC
        return {
            "logit": anomaly_score, # 已经是 detached 的
            "label": batch["label"]
        }