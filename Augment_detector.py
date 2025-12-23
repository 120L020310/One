import argparse
from sklearn.metrics import auc, roc_curve
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import warnings

from Oneclass_XLSR_lit_data_aug import ALDA_OneClass_AugImmunity_Lit
from test import TestTimeAugmentor

# 1. 屏蔽 Python 标准警告
warnings.filterwarnings("ignore")

# 引入你的配置和数据加载 (保持原样)
from Data_Aug import TransformerHardAugmentor
from Oneclass_XLSR_lit_wepe import ALDA_OneClass_Wepe_Lit 
from config.config import get_cfg_defaults
from data.make_dataset import make_data

# ==========================================
# 检测器类
# ==========================================
class Augments_Detector:
    def __init__(self, model, device='cuda'):
        self.device = device
        self.model = model
        
        # 1. 加载模型
        self.model.to(device)
        self.model.eval()
        self.model.freeze() # 冻结所有参数
        
        # 提取 backbone
        self.backbone = self.model.model
        
        # 获取训练好的中心点
        self.centroid = self.model.centroid
        if self.centroid is None:
            # 尝试从 loss_fn 获取，防止 ckpt 保存位置不同
            if hasattr(self.model.loss_fn, 'centroid') and self.model.loss_fn.centroid is not None:
                self.centroid = self.model.loss_fn.centroid
            else:
                raise ValueError("Error: Centroid is None! 模型未正确保存中心点。")
        
        self.centroid = self.centroid.to(device)
        print(f"[Detector] Model loaded. Centroid Shape: {self.centroid.shape}")
        
        # 2. 初始化数据增强器 (用于生成 Noisy View)
        # 注意：这里的参数可以调节。
        # 如果训练时用了 mask_ratio=0.15，测试时用 0.15 是最公平的对比。
        self.augmentor = TransformerHardAugmentor(mask_ratio=0.42, span_len=9600).to(device)
        # self.augmentor = TestTimeAugmentor()

    def predict_batch(self, audio):
        """
        计算单批次数据的异常分数
        """
        # 格式调整 [B, L]
        if len(audio.shape) == 3:
            audio = audio[:, 0, :]
        audio = audio.to(self.device)

        # -------------------------------------------
        # Step 1: Clean Pass (正常特征)
        # -------------------------------------------
        with torch.no_grad():
            res_clean = self.backbone(audio)
            clean_feat = res_clean["final_feat"]
            # 归一化 (非常重要，Cosine Similarity的前提)
            clean_norm = F.normalize(clean_feat, p=2, dim=1)

        # -------------------------------------------
        # Step 2: Noisy Pass (数据增强特征)
        # -------------------------------------------
        # [关键修改] 使用 Augmentor 对 Input Audio 进行物理破坏
        # 而不是使用 get_perturbed_state_dict 对权重进行攻击
        with torch.no_grad():
            audio_noisy = self.augmentor(audio) # [B, L]
            # 使用同一个 backbone 提取特征
            res_noisy = self.backbone(audio_noisy)
            noisy_feat = res_noisy["final_feat"]
            noisy_norm = F.normalize(noisy_feat, p=2, dim=1)

        # -------------------------------------------
        # Step 3: 计算分数
        # -------------------------------------------
        
        # A. Consistency Score (差异度)
        # 我们的假设：
        # Real: clean 和 noisy 很像 -> sim 高 -> (1-sim) 低
        # Fake: clean 和 noisy 差异大 -> sim 低 -> (1-sim) 高
        sim_consistency = (clean_norm * noisy_norm).sum(dim=1)
        score_consistency = 1.0 - sim_consistency
        
        # B. Cluster Score (中心距离)
        # Real: 靠近 centroid -> sim 高 -> (1-sim) 低
        centroid_norm = F.normalize(self.centroid, p=2, dim=0)
        sim_cluster = torch.matmul(clean_norm, centroid_norm)
        score_cluster = 1.0 - sim_cluster
        
        # C. Final Score (融合)
        # 建议：给 Consistency 更高的权重，因为它不仅包含了是否像人，
        # 还包含了 "特征是否结实" 的信息。
        # 比如 final = 1.0 * Cluster + 1.0 * Consistency
        # final_score = 5.0 * score_cluster + 1.0 * score_consistency
        final_score = score_consistency - score_cluster        
        return final_score, score_cluster, score_consistency

    def evaluate_dataloader(self, dataloader, dataset_name="Test"):
        all_scores = []
        all_labels = []
        all_cluster_s = []
        all_consist_s = []
        
        print(f"[Detector] Starting evaluation on {dataset_name} ({len(dataloader)} batches)...")
        idx = 0
        for batch in tqdm(dataloader):
            audio = batch["audio"]
            labels = batch["label"] 
            idx+=1
            final_s, cluster_s, consist_s = self.predict_batch(audio)
            
            all_scores.extend(final_s.cpu().numpy())
            all_cluster_s.extend(cluster_s.cpu().numpy())
            all_consist_s.extend(consist_s.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            if idx==100:break
        return self.calculate_metrics(
            np.array(all_scores), 
            np.array(all_labels),
            np.array(all_cluster_s),
            np.array(all_consist_s)
        )

    def calculate_metrics(self, scores, labels, cluster_scores, consist_scores):
        # target_labels: 1 for Anomaly(Fake), 0 for Normal(Real)
        # 原始 labels: 1=Real, 0=Fake
        target_labels = 1 - labels 
        
        # EER / AUC 计算
        fpr, tpr, thresholds = roc_curve(target_labels, scores, pos_label=1)
        roc_auc = auc(fpr, tpr)
        fnr = 1 - tpr
        eer_idx = np.nanargmin(np.absolute((fnr - fpr)))
        eer = fpr[eer_idx]
        
        # 统计分析
        real_mask = (labels == 1)
        fake_mask = (labels == 0)
        
        print("\n" + "="*50)
        print(f" >>> Performance Report <<<")
        print(f" [Metric] EER  : {eer * 100:.4f}%")
        print(f" [Metric] AUC  : {roc_auc * 100:.4f}%")
        print("-" * 30)
        print(" [Score Analysis - Expect Real < Fake]")
        print(f" Total Score (Real) : {scores[real_mask].mean():.4f}")
        print(f" Total Score (Fake) : {scores[fake_mask].mean():.4f}")
        print("-" * 30)
        print(" [Component Breakdown]")
        print(f" Cluster Dist (Real): {cluster_scores[real_mask].mean():.4f}")
        print(f" Cluster Dist (Fake): {cluster_scores[fake_mask].mean():.4f}")
        print(f" Consistency  (Real): {consist_scores[real_mask].mean():.4f}  <-- 重点关注")
        print(f" Consistency  (Fake): {consist_scores[fake_mask].mean():.4f}  <-- 应该显著变大")
        print("="*50 + "\n")
        
        return {"EER": eer*100, "AUC": roc_auc*100}

# ==========================================
# Main Execution
# ==========================================
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="GMM")
    parser.add_argument("--test_noise", type=int, default=0)
    parser.add_argument("--gpu", type=int, nargs="+", default=0)
    parser.add_argument("-v", "--version", type=int, default=None)
    parser.add_argument("-m", "--model", type=str, default="AASIST")
    parser.add_argument("-ckpt", "--checkpoint", type=str, default=None)
    parser.add_argument("-nr","--noise_ratio",type=float,default=0.5)
    parser.add_argument("--method",type=str,default="sim")
    args = parser.parse_args()
    
    cfg = get_cfg_defaults("config/experiments/%s.yaml" % args.cfg)
    cfg.DATASET.batch_size = 64
    device = torch.device(f'cuda:{args.gpu[0]}' if torch.cuda.is_available() else 'cpu')
    
    # model = MODEL_REGISTRY[args.model]().to(device)
    model = ALDA_OneClass_AugImmunity_Lit()
    ckpt = "/home/zyz/data/test_controlled_ex/1222_XLSR_data_aug_DigitalArtifact_tipletloss/MultiView/ASV2021_LA/version_0/checkpoints/best-epoch=6-val-eer=0.1667.ckpt"
    sd = torch.load(ckpt, map_location="cpu")["state_dict"]
    print(f"Load checkpoint from {ckpt}...")
    model.load_state_dict(sd)
    model = model.to(device=device)
    ds, dl = make_data(cfg.DATASET, args=args)

    # 5. 初始化检测器
    detector = Augments_Detector(model=model, device=device)
    
    # 6. 运行评估
    # 假设 dl.test 是一个列表，包含多个测试集 (ASV-LA, ASV-DF 等)
    for i, test_loader in enumerate(dl.test):
        detector.evaluate_dataloader(test_loader, dataset_name=f"TestSet_{i}")