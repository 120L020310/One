import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.func import functional_call
from models.SLSforASVspoof.teacher import XLSR_Teacher
from myutils.torch.deepfake_detection.audio import DeepfakeAudioClassification
from myutils.zyz.aocloss import AOCloss, AOCloss_plus

import torch

def get_perturbed_state_dict(model, noise_ratio=0.1, target_layer_indices=None, mode='multiplicative'):
    params = dict(model.named_parameters())
    buffers = dict(model.named_buffers())
    perturbed_params = {}
    
    # 定义需要攻击的自定义模块名称 (这是你模型里的变量名)
    # 攻击这些层对结果影响最大！
    custom_target_modules = ["layer_weighting", "temporal_refiner", "projector"]
    
    for name, param in params.items():
        # 跳过 LayerNorm 的 bias
        if "bias" in name: 
            perturbed_params[name] = param
            continue
            
        should_perturb = False
        
        # 1. 检查是否在自定义的关键层中 (Adapter)
        for module_name in custom_target_modules:
            if module_name in name and "bias" not in name and "layer_norm" not in name:
                should_perturb = True
                break
        
        # 2. 如果不在自定义层，再检查是否在目标 Backbone 层中
        if not should_perturb:
            for layer_idx in target_layer_indices:
                if f"layers.{layer_idx}." in name and "weight" in name:
                    should_perturb = True
                    break
        
        # --- 注入噪声 ---
        if should_perturb and 'weight' in name: # 只攻击 weight
            if mode == 'multiplicative':
                noise = torch.randn_like(param) * noise_ratio
                perturbed_params[name] = param * (1 + noise)
            else:
                std = param.std().detach()
                effective_std = std if std > 1e-6 else 1e-3
                noise = torch.randn_like(param) * effective_std * noise_ratio
                perturbed_params[name] = param + noise 
        else:
            perturbed_params[name] = param
            
    return {**perturbed_params, **buffers}
class ALDA_OneClass_Wepe_Lit(DeepfakeAudioClassification):
    def __init__(self, cfg=None, args=None, **kwargs):
        super().__init__()
        
        self.cfg = cfg
        self.args = args
        
        # 1. 骨干网络
        self.model = XLSR_Teacher(proj=True)
        # 2. 特征维度 (XLS-R通常是1024，需根据ALDA_Teacher实际输出调整)
        self.embed_dim = 256 
        
        # 3. 初始化Loss和优化器
        self.configure_loss_fn()
        self.configure_normalizer()
        self.register_buffer("centroid", torch.zeros(self.embed_dim))
        
        # 注意: 单类学习通常不需要最后的 fc 分类层 (self.cls_h)，直接使用特征
        # 但为了特征降维或适配，也可以加一个投影层，这里暂时略去，直接用backbone输出
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
        
        # 带噪的前向传播
        noisy_state_dict = get_perturbed_state_dict(
            self.model, 
            noise_ratio=0.1,  # 建议设大一点，比如 0.1 或 0.15
            target_layer_indices=range(12, 24), # 攻击后 12 层
            mode="multiplicative"
        )
            
        # 使用 functional_call 执行带噪推理
        res_noisy = functional_call(self.model, noisy_state_dict, args=(audio,), kwargs={})
        feat_noisy = res_noisy["final_feat"]
        
        
        
        # 2. 计算 Logit (基于与中心点的相似度)
        if self.loss_fn.centroid == None:
            self.loss_fn.centroid = self.centroid
        centroid = self.loss_fn.centroid
        
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
            "feat_noisy": feat_noisy 
        }
    def on_train_epoch_end(self):
        self.centroid = self.loss_fn.centroid
    def training_step(self, batch, batch_idx,stage="train"):
        batch_res = self._shared_eval_step(batch, batch_idx, stage="train")
        
        # 1. Anchor (Clean) -> Detach
        clean_feat = batch_res["final_feat"]
        clean_norm = F.normalize(clean_feat, p=2, dim=1).detach()
        
        # 2. View (Noisy)
        feat_noisy = batch_res["feat_noisy"]
        noisy_norm = F.normalize(feat_noisy, p=2, dim=1)
        
        # 3. Consistency Loss
        similarity = (clean_norm * noisy_norm).sum(dim=1)
        loss_consistency = (1 - similarity).mean()
        
        # 4. Combine
        loss_cluster = batch_res["loss"]
        loss = 10.0 * loss_cluster + 1.0 * loss_consistency 
        
        self.log("train/loss_cluster", loss_cluster)
        self.log("train/loss_consistency", loss_consistency)
        
        batch_res["loss"] = loss
        if not isinstance(loss, dict):
            loss = {"loss": loss}

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
    