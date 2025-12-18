import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForPreTraining

from models.SLSforASVspoof.teacher import ALDA_Teacher
from myutils.torch.deepfake_detection.audio import DeepfakeAudioClassification
from student import SimpleStudent

class XLS_R_ALDA_lit(DeepfakeAudioClassification):
    
    def __init__(self, cfg=None, args=None, teacher_ckpt_path = None, **kwargs):
        super().__init__()
        
        self.cfg = cfg
        self.args = args
        
        # 1. 初始化模型
        self.teacher = ALDA_Teacher()
        self.student = SimpleStudent(output_dim=1024)
        if teacher_ckpt_path:
            self.load_pretrained_teacher(teacher_ckpt_path)
        # [关键修正] 强制冻结 Teacher 的 Backbone (XLS-R)，只训练附加模块
        self.teacher.eval()
        # for param in self.teacher.xlsr.parameters():
        #     param.requires_grad = False
        for param in self.teacher.parameters():
            param.requires_grad = False
            
        self.configure_loss_fn()
        self.configure_normalizer()
        self.threshold = 0.15 
        self.scale = 10.0
    def load_pretrained_teacher(self, ckpt_path):
        print(f"Loading teacher weights from {ckpt_path} ...")
        
        # 加载 checkpoint (在 CPU 上加载以节省显存)
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        state_dict = checkpoint["state_dict"]
        
        # 创建一个新的字典，用于存放清洗后的权重
        teacher_state_dict = {}
        
        for key, value in state_dict.items():
            # 筛选：只加载属于 'model' 部分的权重
            # 原 ckpt 中的 key 格式为: "model.layer_name.weight"
            if key.startswith("model."):
                # 去掉前缀 "model."，变成 "layer_name.weight"
                # 这样才能匹配 self.model (ALDA_Teacher 类) 内部的参数名
                new_key = key.replace("model.", "", 1)
                teacher_state_dict[new_key] = value
        
        # 将清洗后的权重加载到 self.model 中
        # strict=True 保证所有 ALDA_Teacher 的参数都被正确覆盖
        missing, unexpected = self.teacher.load_state_dict(teacher_state_dict, strict=True)
        
        if len(missing) == 0 and len(unexpected) == 0:
            print("Success: Teacher model weights loaded perfectly.")
        else:
            print(f"Warning: Loading finished with missing: {len(missing)}, unexpected: {len(unexpected)}")
        
    def configure_loss_fn(self):
        # 使用余弦相似度作为重构损失的基础
        self.sim_metric = nn.CosineSimilarity(dim=2)
        
    def configure_normalizer(self):
        # 可以在这里添加 Z-Score Normalization 逻辑 (针对 Anomaly Score)
        pass
        
    def configure_optimizers(self):
        # 参数分组：
        # 1. Teacher Adapter (Layer weights + Time attention) -> 学习率可以稍小
        teacher_adapter_params = list(self.teacher.layer_weighting.parameters()) + \
                                 list(self.teacher.temporal_refiner.parameters())
        
        # 2. Student (Encoder) -> 正常学习率
        student_params = list(self.student.parameters())
        
        optimizer = torch.optim.Adam([
            {'params': student_params, 'lr': 1e-4},
            {'params': teacher_adapter_params, 'lr': 1e-5}
        ], weight_decay=1e-4)
        
        return optimizer

    def calcuate_loss(self, batch_res, batch):
        """
        计算加权重构损失
        注意：训练时只应传入 Real 数据 (label=0)
        """
        t_out = batch_res["teacher_out"]
        s_out = batch_res["student_out"]
        
        t_feat = t_out['final_feat']
        time_weights = t_out['time_weights'] # (B, 1, T)
        
        # 维度对齐 (Student 和 Teacher 由于卷积 padding 可能存在细微长度差异)
        min_len = min(t_feat.shape[1], s_out.shape[1])
        t_feat = t_feat[:, :min_len, :]
        s_out = s_out[:, :min_len, :]
        time_weights = time_weights[:, :, :min_len].squeeze(1)
        
        # 1. 计算余弦相似度 (Sim 越高越好，接近 1)
        sim = self.sim_metric(t_feat, s_out) # (B, T)
        
        # 2. Loss = 1 - Sim (最小化差异)
        loss_map = 1 - sim
        
        # 3. Attention-Guided Loss
        # 只让 Student 学习 Teacher 认为重要的帧 (time_weights 高的区域)
        weighted_loss = loss_map * time_weights
        
        # 归一化 Loss
        loss = weighted_loss.sum() / (time_weights.sum() + 1e-8)
        
        return loss

    def _shared_pred(self, batch, batch_idx, stage='train', **kwargs):
        audio = batch["audio"]
        if len(audio.shape) == 3:
            audio = audio[:, 0, :] # Ensure (B, T)

        # 1. Forward
        # 确保 Teacher Backbone 始终处于 Eval 模式
        self.teacher.xlsr.eval()
        
        t_out = self.teacher(audio)
        s_out = self.student(audio)
        
        # 2. 计算异常分数 (推理逻辑)
        t_feat = t_out['final_feat']
        time_weights = t_out['time_weights'] # (B, 1, T)
        
        # 维度对齐
        min_len = min(t_feat.shape[1], s_out.shape[1])
        t_feat = t_feat[:, :min_len, :]
        s_out = s_out[:, :min_len, :]
        time_weights = time_weights[:, :, :min_len].squeeze(1)
        
        # 计算差异图
        sim = self.sim_metric(t_feat, s_out)
        diff_map = 1 - sim # (B, T)
        
        # Anomaly Score 计算
        # 策略：加权平均差异。分数越高 -> 差异越大 -> 越可能是 Fake
        anomaly_score = (diff_map * time_weights).sum(dim=1) / (time_weights.sum(dim=1) + 1e-8)
        
        # 3. 构造输出
        # anomaly_score 是一个正数，通常 fake 的分数显著高于 real
        # 我们将其作为 Logit (虽然它不是概率的 log，但在 AUC 计算中只看相对大小)
        out = (self.threshold - anomaly_score) * self.scale
        
        # 您的评判函数：
        # Logit > 0 -> Sigmoid > 0.5 -> +0.5 > 1.0 -> int 1 (Real)
        # Logit < 0 -> Sigmoid < 0.5 -> +0.5 < 1.0 -> int 0 (Fake)
        batch_pred = (torch.sigmoid(out) + 0.5).int()
        
        return {
            "logit": out,         # 用于计算 AUC
            "pred": batch_pred,
            "teacher_out": t_out, # 用于计算 Loss
            "student_out": s_out, # 用于计算 Loss
            "feature": t_feat     # 可视化用
        }

    def on_train_epoch_start(self):
        """
        Warm-up 策略：防止冷启动时 Teacher 被带偏
        前 5 个 Epoch 冻结 Teacher Adapter，只让 Student 适应。
        """
        if self.current_epoch < 1:
            # 冻结 Teacher Adapter
            for param in self.teacher.layer_weighting.parameters():
                param.requires_grad = False
            for param in self.teacher.temporal_refiner.parameters():
                param.requires_grad = False
        else:
            # 解冻，开始协同训练
            for param in self.teacher.layer_weighting.parameters():
                param.requires_grad = True
            for param in self.teacher.temporal_refiner.parameters():
                param.requires_grad = True