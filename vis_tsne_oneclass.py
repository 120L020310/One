import argparse
import warnings
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d
warnings.filterwarnings("ignore")

# ==========================================
# 1. 导入你的项目依赖
# ==========================================
from Oneclass_XLSR_lit_data_aug import ALDA_OneClass_AugImmunity_Lit
from config import get_cfg_defaults
from data.make_dataset import make_data
from Oneclass_XLSR_lit_wepe import ALDA_OneClass_Wepe_Lit 

# ==========================================
# 2. 核心计算函数
# ==========================================
def compute_eer(y_true, y_score):
    """
    计算 EER 和 AUC
    y_true: 0 (Real/Normal), 1 (Fake/Anomaly)
    y_score: 异常分数 (越高越像 Fake)
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    roc_auc = auc(fpr, tpr)

    # 计算 EER: FPR == FNR (1-TPR) 的点
    try:
        eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    except Exception:
        # 备用方案：索引查找
        fnr = 1 - tpr
        abs_diffs = np.abs(fpr - fnr)
        min_index = np.argmin(abs_diffs)
        eer = (fpr[min_index] + fnr[min_index]) / 2
        
    return eer, roc_auc

def extract_features(model, dataloader, device, max_batches=None):
    """
    提取测试集特征和标签
    """
    model.eval()
    all_feats = []
    all_labels = []
    
    print("Extracting features...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader)):
            audio = batch["audio"].to(device)
            label = batch["label"] # 1=Real, 0=Fake
            
            if len(audio.shape) == 3:
                audio = audio[:, 0, :]
                
            # 1. 前向传播
            res = model.model(audio)
            feat = res["final_feat"]
            
            # 2. 关键：归一化
            feat_norm = F.normalize(feat, p=2, dim=1)
            
            all_feats.append(feat_norm.cpu().numpy())
            all_labels.append(label.numpy())
            
            if max_batches is not None and i >= max_batches:
                break
                
    # 拼接数据
    all_feats = np.concatenate(all_feats, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    
    return all_feats, all_labels

def visualize_tsne(features, labels, centroid, save_path="tsne_oneclass.png"):
    """
    绘制 t-SNE：Real vs Fake vs Centroid
    """
    print(f"Computing t-SNE (Total samples: {len(labels)})...")
    
    # 1. 数据采样
    n_samples = 1000
    real_idx = np.where(labels == 1)[0]
    fake_idx = np.where(labels == 0)[0]
    
    if len(real_idx) > n_samples:
        real_idx = np.random.choice(real_idx, n_samples, replace=False)
    if len(fake_idx) > n_samples:
        fake_idx = np.random.choice(fake_idx, n_samples, replace=False)
        
    sampled_idx = np.concatenate([real_idx, fake_idx])
    
    X_sampled = features[sampled_idx]
    y_sampled = labels[sampled_idx]
    
    # 2. 处理 Centroid
    centroid_np = centroid.cpu().numpy().reshape(1, -1)
    centroid_norm = centroid_np / np.linalg.norm(centroid_np)
    
    # 3. 合并数据进行 t-SNE 降维
    X_all = np.concatenate([centroid_norm, X_sampled], axis=0)
    
    # t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X_all)
    
    # 4. 拆分坐标
    center_2d = X_embedded[0]
    data_2d = X_embedded[1:]
    
    # 5. 绘图
    plt.figure(figsize=(10, 8))
    
    # 绘制 Real (Label=1)
    plt.scatter(
        data_2d[y_sampled == 1, 0], 
        data_2d[y_sampled == 1, 1], 
        c='green', label='Real (Bonafide)', alpha=0.6, s=15, edgecolors='none'
    )
    
    # 绘制 Fake (Label=0)
    plt.scatter(
        data_2d[y_sampled == 0, 0], 
        data_2d[y_sampled == 0, 1], 
        c='red', label='Fake (Spoof)', alpha=0.4, s=15, edgecolors='none'
    )
    
    # 绘制 Centroid
    plt.scatter(
        center_2d[0], center_2d[1], 
        c='black', label='Centroid', s=300, marker='*', edgecolors='white', linewidth=1.5
    )
    
    plt.title("t-SNE Visualization of One-Class Clustering")
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Saved visualization to {save_path}")
def plot_score_histogram(scores, labels,save_path):
    real_scores = scores[labels == 1] # 假设 Label 1 是 Real
    fake_scores = scores[labels == 0] # 假设 Label 0 是 Fake
    
    plt.figure(figsize=(10, 6))
    plt.hist(real_scores, bins=50, alpha=0.5, label='Real (Bonafide)', color='green', density=True)
    plt.hist(fake_scores, bins=50, alpha=0.5, label='Fake (Spoof)', color='red', density=True)
    plt.xlabel('Anomaly Score (Distance to Centroid)')
    plt.ylabel('Density')
    plt.title('Score Distribution Histogram')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
# ==========================================
# 3. 主程序
# ==========================================
if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="GMM")
    parser.add_argument("--gpu", type=int, nargs="+", default=0)
    parser.add_argument("-v", "--version", type=int, default=None)
    parser.add_argument("--test_noise", type=int, default=0)
    parser.add_argument("-nr","--noise_ratio",type=float,default=0.1) 
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("-ckpt", "--checkpoint", type=str, 
                        default="/home/zyz/data/test_controlled_ex/1222_XLSR_data_aug_baseline_TransformerHardAugmentor_0.1_4800/MultiView/ASV2021_LA/version_0/checkpoints/best-epoch=3-val-eer=0.1085.ckpt")
    args = parser.parse_args()
    
    # 1. 配置环境
    cfg = get_cfg_defaults("config/experiments/%s.yaml" % args.cfg)
    cfg.DATASET.batch_size = args.batch_size
    device = torch.device(f'cuda:{args.gpu[0]}' if torch.cuda.is_available() else 'cpu')
    
    # 2. 加载模型
    print(f"Loading model from {args.checkpoint} ...")
    model = ALDA_OneClass_AugImmunity_Lit(proj=True)
    
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    sd = checkpoint["state_dict"]
        
    model.load_state_dict(sd, strict=False)
    model = model.to(device)
    model.eval()
    
    # 3. 检查 Centroid
    if hasattr(model, "centroid"):
        centroid = model.centroid
    elif hasattr(model.loss_fn, "centroid"):
        centroid = model.loss_fn.centroid
    else:
        print("Warning: Centroid not found, initializing zeros.")
        centroid = torch.zeros(model.embed_dim).to(device)

    print(f"Centroid loaded. Norm: {centroid.norm().item():.4f}")

    # 4. 准备数据
    ds, dl = make_data(cfg.DATASET, args=args)
    test_dataloaders = dl.test if isinstance(dl.test, list) else [dl.test]
    
    # 5. 遍历测试集
    for i, test_loader in enumerate(test_dataloaders):
        print(f"\nProcessing Test Set {i}...")
        
        # -----------------------------------------------------
        # [修改点] 为了计算准确的 AUC/EER，最好使用全量数据
        # 如果你只想调试看一眼，保留 max_batches=50
        # 如果要发论文或看真实效果，请改为 max_batches=None
        # -----------------------------------------------------
        feats, labels = extract_features(model, test_loader, device, max_batches=50) 
        
        # ==========================================
        # [新增] 计算 AUC 和 EER
        # ==========================================
        # 1. 准备 Centroid (numpy 格式并归一化)
        centroid_np = centroid.cpu().numpy().reshape(1, -1)
        centroid_norm = centroid_np / np.linalg.norm(centroid_np)
        
        # 2. 计算分数 (Distance)
        # Cosine Similarity: (N, D) * (D, 1) -> (N, 1)
        sims = np.dot(feats, centroid_norm.T).squeeze() # Shape: (N,)
        # Anomaly Score: 距离越远(相似度越低)，分数越高(越像Fake)
        scores = 1 - sims 
        # plot_score_histogram(scores, labels,f"best_xlsr/zhifangtu{i}_auc{auc_score*100:.2f}.png")
        # 3. 准备标签 (转换为异常检测标准)
        # 原始: Real=1, Fake=0
        # 目标: Normal(Real)=0, Anomaly(Fake)=1
        y_true_anomaly = 1 - labels
        
        # 4. 计算指标
        eer, auc_score = compute_eer(y_true_anomaly, scores)
        
        print(f"------ Test Set {i} Metrics ------")
        print(f"  > AUC : {auc_score*100:.2f}%")
        print(f"  > EER : {eer*100:.2f}%")
        print(f"----------------------------------")

        # 绘图 (文件名加上 AUC 以便区分)
        save_name = f"best_xlsr/tsne_set{i}_auc{auc_score*100:.2f}.png"
        visualize_tsne(feats, labels, centroid, save_path=save_name)