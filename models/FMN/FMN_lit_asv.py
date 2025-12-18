import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import roc_auc_score
import torch.nn.functional as F

# 引入原本的工具函数
from FMN import (
    find_best_threshold, 
    freeze_adapters, freeze_fmn, normal_margin_loss, 
    unfreeze_adapters, unfreeze_fmn, zero_init_adapters
)
from myutils.datasets.audio.Augment import ASV2021_MLAAD_Augmentor, PseudoAnomalyAugmentor
from myutils.torch.deepfake_detection.base import BinaryClassification
from myutils.zyz.my_utils import LabelSmoothingBCE

class DFM_PL(BinaryClassification):
    def __init__(
        self,
        model,
        fmn,
        num_domains=4,
        da_lambda=0.0,
        target_layer=[6],
        adapter_lr=5e-5,
        fmn_lr=1e-4,
        stage1_epochs=10,
        stage2_epochs=0,
        weight_decay=1e-6,
        clip_grad_norm=5.0
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['model', 'fmn'])
        self.model = model
        self.fmn = fmn
        self.automatic_optimization = False
        
        self.asv2021_mlaad_augmenter = ASV2021_MLAAD_Augmentor(sample_rate=16000)
        
        # Stage 1 Loss (手动计算不用这个对象，但保留定义)
        self.cosine_loss = nn.CosineEmbeddingLoss(margin=0.5)
        
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.register_buffer("energy_threshold_map", torch.tensor([0.1, 0.2, 0.4, 0.5]))   
        self.bce_loss = LabelSmoothingBCE(label_smoothing=0.1)    
        
        # 初始化 Adapter 为 0
        zero_init_adapters(self.model)

    def configure_optimizers(self):
        adapter_params = [p for p in self.model.adapters.parameters() if p.requires_grad]
        opt_adapter = torch.optim.Adam(adapter_params, lr=self.hparams.adapter_lr, weight_decay=self.hparams.weight_decay)

        fmn_params = [p for p in self.fmn.parameters() if p.requires_grad]
        opt_fmn = torch.optim.Adam(fmn_params, lr=self.hparams.fmn_lr, weight_decay=self.hparams.weight_decay)

        return [opt_adapter, opt_fmn]

    def _shared_pred(self, batch, stage="train"):
        wave = batch['audio']
        if stage != "train":
            self.model.eval(); self.fmn.eval()
            
        emb = self.model(wave, target_layer=self.hparams.target_layer) # (B, T, D)
        
        if self.get_current_stage() == "stage2":
            scores, score_patch, score_global = self.fmn(emb)
            return {
                "fmn_scores": scores,
                "emb": emb,
                "logit": -scores
            }
        else:
            return {"emb": emb}
    def configure_loss_fn(
        self,
    ):
        self.bce_loss = LabelSmoothingBCE(label_smoothing=0.1)

    def calcuate_loss(self, batch_res, batch):
        """Stage 2 FMN Loss"""
        scores = batch_res["logit"]
        label = batch["label"].to(scores.device)
        loss = self.bce_loss(scores, label)
        # real_mask = (labels == 1)
        # fake_mask = (labels == 0)
        
        # loss = 0.0
        # if real_mask.sum() > 0:
        #     loss += normal_margin_loss(scores[real_mask], margin_n=0.05)
        # if fake_mask.sum() > 0:
        #     loss += torch.clamp(0.8 - scores[fake_mask], min=0).mean()

            
        return loss

    def get_current_stage(self):
        cycle_len = self.hparams.stage1_epochs + self.hparams.stage2_epochs
        if cycle_len == 0: return "stage1"
        epoch_in_cycle = self.current_epoch % cycle_len
        return "stage1" if epoch_in_cycle < self.hparams.stage1_epochs else "stage2"

    def on_train_epoch_start(self):
        stage = self.get_current_stage()
        if stage == "stage1":
            freeze_fmn(self.fmn)
            unfreeze_adapters(self.model)
            self.model.train()
            self.fmn.eval()
            print(f"\n[Epoch {self.current_epoch}] Stage 1: Adapter Training (Dynamic Coverage Alignment)")
        else:
            freeze_adapters(self.model)
            unfreeze_fmn(self.fmn)
            self.model.eval()
            self.fmn.train()
            print(f"\n[Epoch {self.current_epoch}] Stage 2: FMN Training (Scoring Strategy)")

    def training_step(self, batch, batch_idx, dataloader_idx=0):
        stage_name = self.get_current_stage()
        opt_adapter, opt_fmn = self.optimizers()
        
        wave = batch['audio']
        labels = batch['label'] if "label" in batch else torch.ones(wave_real.size(0), device=wave_real.device)
        # 2. 全局前向传播
        batch_res = self._shared_pred(batch, stage="train")

        # ================= STAGE 1 =================
        if stage_name == "stage1":
            current_opt = opt_adapter
            
            loss_real = torch.tensor(0.0, device=self.device)
            loss_push = torch.tensor(0.0, device=self.device)
            monitor_score_real = torch.tensor([], device=self.device)
            monitor_score_fake = torch.tensor([], device=self.device)
            
            _ = self.fmn.bank_thresholds # 确保加载
            
            # --- 【新增】初始化收集容器 ---
            collected_logits = []
            collected_labels = []
            real_idx = (labels == 1)
            # --- A. Real Loss (Mean + Top-K Hard Mining) --------------------------------------------------------
            # 获取 Real 数据
            emb_real = batch_res["emb"][real_idx]   
            wave_real = wave[real_idx]         
            # 计算能量 Mask
            expected_len = emb_real.shape[1]
            mask_real = self.get_energy_mask(wave_real, target_len=expected_len).to(self.device)
            
            # 1. 计算全量违规图 (B, T) - 【关键：保留梯度用于 Top-K Loss】
            feat_flat_real = emb_real.view(-1, 768)
            violation_flat_real, _ = self.compute_violation_score(feat_flat_real, return_mean=False)
            violation_map_real = violation_flat_real.view(emb_real.shape[0], emb_real.shape[1])
            
            # 2. 应用 Mask (静音区置0)
            active_violation_real = violation_map_real * mask_real

            if mask_real.sum() > 0:
                # --- Part 1: Global Mean Loss (维持基本盘) ---
                # 取出所有有效 Patch
                valid_scores = active_violation_real[mask_real.bool()]
                monitor_score_real = valid_scores.detach()
                
                # 基础 Loss: 平均违规程度
                loss_mean = torch.relu(valid_scores + 0.05).mean()
                
                # --- Part 2: Top-K Hard Mining Loss & Logit Collection ---
                # 我们在这个循环里同时做两件事：
                # 1. 找出每个样本最难的 Patch 计算额外的 Loss (对齐测试逻辑)
                # 2. 收集这些 Top-K 分数作为 Logit 给 Callback
                real_topk_losses = []
                
                B_real = wave_real.size(0)
                for i in range(B_real):
                    curr_mask = mask_real[i].bool()
                    if curr_mask.sum() == 0:
                        collected_logits.append(0.0) # 完美 Real
                    else:
                        v_patches = active_violation_real[i][curr_mask]
                        
                        # Top-K 聚合 (策略与 Fake/Test 保持一致)
                        # 例如: 关注最难的 20% 或 50%
                        k = max(1, int(len(v_patches) * 0.5)) 
                        topk_v, _ = torch.topk(v_patches, k)
                        
                        # 核心分数 (Scalar Tensor, 带梯度)
                        sample_score = topk_v.mean()
                        
                        # a. 计算 Top-K Loss (针对难样本的额外惩罚)
                        real_topk_losses.append(torch.relu(sample_score + 0.05))
                        
                        # b. 收集 Logit (越高越真 -> 负数)
                        collected_logits.append(-sample_score.item())
                        
                    collected_labels.append(1) # Label = Real
                
                # 聚合 Real Loss
                if len(real_topk_losses) > 0:
                    loss_topk = torch.stack(real_topk_losses).mean()
                else:
                    loss_topk = torch.tensor(0.0, device=self.device)
                
                # 【最终 Real Loss】 = 平均 Loss + 难样本 Top-K Loss
                loss_real = loss_mean + loss_topk

            else:
                # 全静音 Batch 处理
                for _ in range(wave_real.size(0)):
                    collected_logits.append(0.0)
                    collected_labels.append(1)

            # --- B. Fake Loss (样本级: 累积违规量足够大即可) ---------------------------------------------------------------
            fake_idx = (labels == 0)
            wave_fake = wave[fake_idx]
            # 1. 提取特征 (B, T, D)
            emb_fake = self.model(wave_fake.squeeze(0), target_layer=self.hparams.target_layer)
            B_fake, T_fake, D_fake = emb_fake.shape
            
            # 2. 对齐 Mask
            expected_len = emb_fake.shape[1]
            
            # 【关键】使用对应的能量阈值
            eng_mask = self.get_energy_mask(wave_fake.squeeze(0), target_len=expected_len).to(self.device)
            
            # 有效 Mask = 被篡改 AND 有能量
            valid_mask = eng_mask 
            
            if valid_mask.sum() > 0:
                # 3. 计算所有 Patch 的违规分
                feat_flat = emb_fake.view(-1, D_fake)
                violation_flat, _ = self.compute_violation_score(feat_flat, return_mean=False)
                violation_map = violation_flat.view(B_fake, T_fake)
                
                # 只看 Mask 区域
                active_violation = violation_map * valid_mask
                
                sample_losses = []
                sample_scores = []
                
                # 4. 样本级聚合 (Sample-Level Aggregation)
                for i in range(B_fake):
                    current_mask = valid_mask[i].bool()
                    if current_mask.sum() == 0: continue
                    
                    # 取出该样本所有 Active Patch 的违规分
                    v_patches = active_violation[i][current_mask]
                    
                    # Top-K Mean: 只要最严重的 20% 区域违规分很高，就判为假
                    k = max(1, int(len(v_patches) * 0.2)) 
                    topk_v, _ = torch.topk(v_patches, k)
                    sample_score = topk_v.mean()
                    
                    sample_scores.append(sample_score)
                    
                    # 5. Loss 计算
                    # 目标: Sample Score > 0.2 (违规分要显著)
                    TARGET_SCORE = 0.4 
                    sample_losses.append(torch.relu(TARGET_SCORE - sample_score))

                    # === 【新增】Fake 样本 Logit 收集 ===
                    collected_logits.append(-sample_score.item()) # Logit = -Score
                    collected_labels.append(0) # Label = Fake
                    # ==================================
                
                if len(sample_losses) > 0:
                    loss_push = torch.stack(sample_losses).mean()
                    monitor_score_fake = torch.stack(sample_scores).detach()
            
            # === 【新增】注入 Logits 和 Labels 到 Batch 中供 Callbacks 使用 ===
            if len(collected_logits) > 0:
                # 放入 batch_res 供 Callbacks 的 preds = batch_res["logit"] 读取
                batch_res['logit'] = torch.tensor(collected_logits, device=self.device)
                batch_res['logit'] = batch_res['logit']+0.64
                
                # 修改 batch['label'] 供 Callbacks 的 targets = batch["label"] 读取
                batch['label'] = torch.tensor(collected_labels, device=self.device)
            else:
                batch_res['logit'] = torch.tensor([], device=self.device)
            # ==============================================================
            
            # 兼容性字段 (可选)
            batch_res['pred'] = (torch.sigmoid(batch_res["logit"]) + 0.64).int()

            # --- C. 监控 -------------------------------------------------------------------------------
            score_real_val = monitor_score_real.mean() if monitor_score_real.numel() > 0 else 0.0
            score_fake_val = monitor_score_fake.mean() if monitor_score_fake.numel() > 0 else 0.0
            
            # Gap 越大越好
            gap_val = score_fake_val - score_real_val
            
            self.log_dict({
                "train/loss_real": loss_real,
                "train/loss_push": 2 * loss_push,
                "train/score_real": score_real_val, # 目标 -> 0 (越低越好)
                "train/score_fake": score_fake_val, # 目标 -> >0.1 (越大越好)
                "train/gap": gap_val
            }, prog_bar=True, on_step=True)

            loss = 1.0 * loss_real + 2.0 * loss_push
            
            current_opt.zero_grad()
            if loss.requires_grad and loss > 0.0:
                self.manual_backward(loss)
                self.clip_gradients(current_opt, gradient_clip_val=self.hparams.clip_grad_norm, gradient_clip_algorithm="norm")
                current_opt.step()

        # ================= STAGE 2 =================
        else:
            current_opt = opt_fmn
            loss = self.calcuate_loss(batch_res, batch)
            self.log("train/loss_fmn", loss, on_step=True, prog_bar=True)
            
            current_opt.zero_grad()
            self.manual_backward(loss)
            self.clip_gradients(current_opt, gradient_clip_val=self.hparams.clip_grad_norm, gradient_clip_algorithm="norm")
            current_opt.step()
        # if batch_idx==5:
        #     self.trainer.should_stop=True
        return batch_res
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        wave = batch['audio']
        labels = batch['label']

        # 2. 提取特征
        emb = self.model(wave, target_layer=self.hparams.target_layer)
        B, T, D = emb.shape
        
        # 3. 计算 Energy Mask (过滤静音)
        expected_len = emb.shape[1]
        eng_mask = self.get_energy_mask(wave.squeeze(1), target_len=expected_len).to(self.device)
        
        # 4. 计算所有 Patch 的违规分 (Violation Score)
        # return_mean=False -> 返回 (B, T) 的 map
        feat_flat = emb.view(-1, D)
        violation_flat, _ = self.compute_violation_score(feat_flat, return_mean=False)
        violation_map = violation_flat.view(B, T)
        
        # 5. 应用 Mask (只看有效语音区域)
        # 将静音区域的违规分置为 0 (安全)
        active_violation = violation_map * eng_mask 
        
        # 6. 聚合策略: Sample-Level Logit
        # 对于每一条音频，找出其"最假"的 K 个 Patch，取平均作为该音频的"伪造程度"
        # 即使是 Real 样本，我们也算这个值（Real 的这个值应该很小，接近 0）
        sample_logits = []
        
        k_percent = 0.20 # 关注最严重的 20%
        
        for i in range(B):
            # 取出当前样本的有效 Patch
            current_mask = eng_mask[i].bool()
            
            if current_mask.sum() == 0:
                # 极端情况：全是静音
                # 判为"非常Real" (Logit=0.0 或者稍微给点惩罚，这里给 0 代表不违规)
                sample_logits.append(0.0)
                continue
            
            v_patches = active_violation[i][current_mask]
            
            # Top-K Aggregation
            k = max(1, int(len(v_patches) * k_percent))
            topk_v, _ = torch.topk(v_patches, k)
            
            # 这是 "Anomaly Score" (越高越假)
            anomaly_score = topk_v.mean().item()
            
            # 【关键转换】
            # Logit = -Anomaly Score
            # 越高越真 (Label=1), 越低越假 (Label=0)
            sample_logits.append(-anomaly_score)
            
        # 转为 Tensor
        logits_tensor = torch.tensor(sample_logits, device=self.device)
        logits_tensor = logits_tensor + 0.64
        
        # 存入列表
        self.validation_step_outputs.append((logits_tensor.cpu().numpy(), labels.cpu().numpy()))
        
        return {"logit": logits_tensor, "label": labels}


    def test_step(self, batch, batch_idx, dataloader_idx=0):
        return self.validation_step(batch,batch_idx,dataloader_idx)

    # def on_test_epoch_end(self):
    #     if not self.test_step_outputs: return
    #     preds, labels = [], []
    #     for p, l in self.test_step_outputs:
    #         preds.extend(p)
    #         labels.extend(l)
    #     self.test_step_outputs.clear()
        
    #     scores = np.array(preds)
    #     y_true = np.array(labels)
    #     y_scores = -scores
        
    #     try:
    #         auc = roc_auc_score(1 - y_true, y_scores)
    #     except:
    #         auc = 0.0
    #     best_th, eer, _, _ = find_best_threshold(y_scores, y_true)
        
    #     print(f"\n==== TEST RESULTS ====")
    #     print(f"Real Margin: {scores[y_true==1].mean():.3f} | Fake Margin: {scores[y_true==0].mean():.3f}")
    #     print(f"AUC: {auc:.4f} | EER: {eer:.4f}")
        
    #     self.log_dict({"test/auc": auc, "test/eer": eer})

    def get_energy_mask(self, wave, target_len, threshold=0.01):
        if wave.dim()==3:
            wave = wave.squeeze(1)
        B, T_wave = wave.shape
        stride = T_wave // target_len
        cutoff = target_len * stride
        wave_cut = wave[:, :cutoff]
        wave_reshaped = wave_cut.view(B, target_len, stride)
        energy = wave_reshaped.abs().mean(dim=2)
        return (energy > threshold).float()

    # =========================================================
    #  核心辅助函数：计算最佳覆盖违规分 (Robust Violation Score)
    # =========================================================
    def compute_violation_score(self, emb, return_mean=True):
        """
        计算样本的违规分数。
        逻辑：遍历所有 Bank 中心，看样本是否落在"任意"一个中心的覆盖范围内。
        Violation = ReLU( - Max(Sim - Threshold) )
        """
        emb_flat = emb
            
        # 1. 归一化
        feat_norm = F.normalize(emb_flat, p=2, dim=1)
        bank = self.fmn.memory_bank.to(self.device)
        bank_thresholds = self.fmn.bank_thresholds.to(self.device) # (K,)
        
        # 2. 计算所有点到所有中心的相似度矩阵 (N, K)
        # 注意：这里计算量比之前大了一点，因为没有先 argmax
        sim_matrix = torch.matmul(feat_norm, bank.t())
        
        # 3. 计算所有中心的安全边际 (Safety Margin)
        # Margin = Sim - Threshold
        # 利用广播机制: (N, K) - (1, K)
        margin_matrix = sim_matrix - bank_thresholds.unsqueeze(0)
        
        # 4. 找到"最佳覆盖" (Max Margin)
        # 即使我离 A 最近，但 A 的门槛高导致 margin<0；
        # 而我离 B 稍远，但 B 的门槛低导致 margin>0；
        # max() 会自动选择 B。
        max_margin, best_idx = margin_matrix.max(dim=1) # (N,)
        
        # 5. 计算违规量 (Violation)
        # 如果 max_margin > 0 (在某岛内)，Violation = 0
        # 如果 max_margin < 0 (在所有岛外)，Violation = -max_margin (>0)
        violation = torch.relu(-max_margin)
        
        # 6. 聚合
        return violation, max_margin