import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

# 假设这些是你原本的引用
from models.SLSforASVspoof.teacher import ALDA_Teacher
from myutils.torch.deepfake_detection.audio import DeepfakeAudioClassification
from myutils.zyz.aocloss import AOCloss 

class ALDA_OneClass_Lit(DeepfakeAudioClassification):
    def __init__(self, cfg=None, args=None, **kwargs):
        super().__init__()
        
        self.cfg = cfg
        self.args = args
        
        # 1. 骨干网络
        self.model = ALDA_Teacher()
        
        # 2. 特征维度 (XLS-R通常是1024，需根据ALDA_Teacher实际输出调整)
        self.embed_dim = 1024 
        
        # 3. 初始化Loss和优化器
        self.configure_loss_fn()
        self.configure_normalizer()
        
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

    def calcuate_loss(self, batch_res, batch):
        """
        计算单类损失
        """
        # 获取特征 [Batch, 1024]
        # 使用 mean pooling 将 (B, C, T) -> (B, C)
        embeddings = batch_res["final_feat"]
        
        # ---------------- CRITICAL LOGIC ----------------
        # 你的需求：训练集全是 label=1 的真实音频。
        # 你的AOCloss逻辑：labels==0 是 Bonafide (用于更新centroid)。
        # 因此，在训练计算Loss时，我们需要强制欺骗Loss函数，告诉它这些都是 class 0
        
        if self.training:
            # 训练阶段：强制所有标签为 0 (Bonafide)，以便更新 centroid 并计算距离损失
            # 这样 loss = 1 - sim(embedding, centroid)
            virtual_labels = torch.zeros(embeddings.shape[0], device=embeddings.device, dtype=torch.long)
        else:
            # 验证/测试阶段：使用真实标签计算Loss (如果验证集包含fake数据)
            # 注意：如果你的 validation set label 1 是 fake，这里需要映射逻辑：
            # 假设 dataset: 0=real, 1=fake -> 直接传
            # 假设 dataset: 1=real, 0=fake -> 需要翻转
            # 这里默认遵循 AOCloss 的定义 (0=Bonafide)
            # 如果你的Dataset label=1是Real，这里需要 label = 1 - batch["label"]
            # 既然你提到 "训练集全是label=1的真实音频"，我假设你的Dataset定义是 1=Real。
            # 为了适配 AOCloss (0=Real)，我们需要反转标签。
            input_label = batch["label"].type(torch.long)
            # 将 1(Real) 变为 0(Real for Loss), 0(Fake) 变为 1(Fake for Loss)
            # 这是一个假设，请根据你实际Dataset的label定义修改！
            # 下面这行代码假设: Dataset(1=Real, 0=Fake/Other) -> Loss(0=Real, 1=Fake)
            virtual_labels = 1 - input_label 
            
        loss = self.loss_fn(embeddings, virtual_labels)
        return loss

    def _shared_pred(self, batch, batch_idx, stage='train', **kwargs):
        audio = batch["audio"]
        if len(audio.shape) == 3:
            audio = audio[:, 0, :]

        # 1. 获取模型输出
        res = self.model(audio)
        # 获取 Embedding (B, 1024)
        embedding = res["final_feat"].mean(1)
        
        # 2. 计算 Logit (基于与中心点的相似度)
        centroid = self.loss_fn.centroid
        
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
            
            # 3. 将相似度转换为 Logit
            # 逻辑：
            # Similarity 高 (接近1) -> 是中心(Real) -> 应该是 Class 0 -> Logit 应该很小 (负数)
            # Similarity 低 (接近0或-1) -> 离中心远(Fake) -> 应该是 Class 1 -> Logit 应该很大 (正数)
            # 
            # 简单映射: logit = -similarity
            # 这样: Sim=1 -> Logit=-1 (pred 0); Sim=-1 -> Logit=1 (pred 1)
            # 你也可以用 logit = 1 - similarity (范围 [0, 2])
            logit = similarity-0.49

        # 4. 生成硬预测 (Pred)
        # 这里的阈值选取很重要。
        # 如果 logit = -similarity，由于 sim 均值可能在 0.5 左右，
        # 简单起见，我们可以设阈值为 -threshold (比如 sim < 0.5 判为假)
        # 这里为了保持代码跑通，暂时用 0 作为分界线 (即 similarity > 0 判为真)
        # 在实际部署中，你需要根据 EER 调整这个阈值。
        batch_pred = (logit > 0.95).int() 
        
        # 这里的 output 结构保持和你原来的一致
        return {
            "final_feat": embedding,
            "logit": logit, # 用于计算 EER, AUC 等指标
            "pred": batch_pred, # 硬预测标签
            "embedding": embedding # 额外返回 embedding 用于调试或t-SNE可视化
        }