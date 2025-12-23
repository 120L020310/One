import argparse
import warnings
import numpy as np
from sklearn.metrics import auc, roc_curve
import torch
import torch.nn.functional as F
from tqdm import tqdm
warnings.filterwarnings("ignore")
from Oneclass_XLSR_lit_data_aug import ALDA_OneClass_AugImmunity_Lit
from config.config import get_cfg_defaults
from data.make_dataset import make_data

class Adversarial_WePe_Detector:
    def __init__(self, model, device='cuda', epsilon=0.012):
        """
        epsilon: 攻击力度。建议从 0.001 开始尝试，逐渐增加到 0.01。
                 如果太大，Real 也会被攻破；如果太小，Fake 没反应。
        """
        self.device = device
        self.model = model
        self.model.to(device)
        self.model.eval()
        # 注意：这里不 Freeze 模型，因为我们需要梯度流过模型传回 Input
        # 但我们不会调用 optimizer.step()，所以权重是安全的。
        
        self.backbone = self.model.model
        self.centroid = self.model.centroid.to(device).detach() # 确保 centroid 不带梯度
        self.epsilon = epsilon
        
        print(f"[Adv Detector] Initialized with epsilon={self.epsilon}")

    def predict_batch(self, audio):
        """
        Args:
            audio: [B, L] 原始音频
        """
        if len(audio.shape) == 3:
            audio = audio[:, 0, :]
        
        # 1. 准备 Clean Input (需要计算梯度)
        audio_clean = audio.clone().detach().to(self.device)
        audio_clean.requires_grad = True # <--- 关键：开启输入的梯度需求
        
        # ==========================================
        # Step 1: Forward Pass (Clean)
        # ==========================================
        # 必须显式开启 grad，即使是 eval 模式
        with torch.enable_grad():
            res_clean = self.backbone(audio_clean)
            feat_clean = res_clean["final_feat"]
            norm_clean = F.normalize(feat_clean, p=2, dim=1)
            
            # 计算当前距离中心的 Loss
            # 我们希望攻击能让这个距离变大，即 Similarity 变小
            # Loss = Cosine Similarity (我们要最小化 Sim，或者最大化 1-Sim)
            # 这里的攻击目标是：让样本看起来“不像”中心。
            # 所以 Attack Loss = Similarity (我们对 Sim 求导，梯度指向 Sim 增加的方向)
            # 我们需要沿着梯度 *反方向* 走 (Gradient Descent) 从而减小 Sim？
            # 不，FGSM 是 x + eps * sign(grad)。
            # 如果 Loss = Distance，我们要 Maximize Distance -> x + eps * sign(grad_dist)
            # 如果 Loss = Similarity，我们要 Minimize Similarity -> x - eps * sign(grad_sim)
            
            norm_centroid = F.normalize(self.centroid, p=2, dim=0)
            similarity = torch.matmul(norm_clean, norm_centroid)
            
            # 定义对抗目标：让相似度越小越好（离群）
            # 所以 Loss 定义为 Similarity，我们要 minimize 它。
            loss = similarity.sum()
            
            # 计算梯度：d(Sim)/d(Audio)
            # 这告诉我们：怎么修改 Audio，能让 Sim 变大。
            # 所以我们要往反方向走。
            self.model.zero_grad()
            loss.backward()
            
            # 获取数据梯度
            data_grad = audio_clean.grad.data
        
        # ==========================================
        # Step 2: Generate Adversarial Example
        # ==========================================
        # FGSM: x_adv = x - epsilon * sign(grad_sim)
        # (往相似度降低的方向走)
        with torch.no_grad():
            audio_adv = audio_clean - self.epsilon * data_grad.sign()
            # 也可以截断一下幅度防止爆音，但通常 epsilon 很小不需要
            
        # ==========================================
        # Step 3: Forward Pass (Adversarial)
        # ==========================================
        with torch.no_grad():
            res_adv = self.backbone(audio_adv)
            feat_adv = res_adv["final_feat"]
            norm_adv = F.normalize(feat_adv, p=2, dim=1)
            
        # ==========================================
        # Step 4: Calculate Scores
        # ==========================================
        # 1. Cluster Score (Clean)
        sim_cluster = torch.matmul(norm_clean.detach(), norm_centroid)
        score_cluster = 1.0 - sim_cluster
        
        # 2. Consistency Score (Clean vs Adv)
        # Real: 很难被攻击，Adv 和 Clean 依然很像 -> Sim 高 -> Score 低
        # Fake: 很容易被踢飞，Adv 和 Clean 差异大 -> Sim 低 -> Score 高
        sim_consistency = (norm_clean.detach() * norm_adv).sum(dim=1)
        score_consistency = 1.0 - sim_consistency
        
        # 融合
        final_score = 5.0* score_cluster + 2.0 * score_consistency
        
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
    cfg.DATASET.batch_size = 32
    cfg.DATASET.test_batch_size = 32
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
    detector = Adversarial_WePe_Detector(model=model, device=device)
    
    # 6. 运行评估
    # 假设 dl.test 是一个列表，包含多个测试集 (ASV-LA, ASV-DF 等)
    for i, test_loader in enumerate(dl.test):
        detector.evaluate_dataloader(test_loader, dataset_name=f"TestSet_{i}")