import argparse
import os
import warnings
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from tqdm import tqdm

from Oneclass_XLSR_lit_data_aug_wepe import ALDA_OneClass_AugImmunity_wepe_Lit
warnings.filterwarnings("ignore")

# ==========================================
# 1. 导入你的项目依赖
# ==========================================
from Oneclass_XLSR_lit_data_aug import ALDA_OneClass_AugImmunity_Lit
from Oneclass_XLSR_lit_wepe import ALDA_OneClass_Wepe_Lit
from config.config import get_cfg_defaults
from data.make_dataset import make_data
from get_true_dataset import get_true

# ==========================================
# 2. 核心函数
# ==========================================

def compute_eer(y_true, y_score):
    """
    计算 EER 和 AUC
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    roc_auc = auc(fpr, tpr)
    try:
        eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    except Exception:
        fnr = 1 - tpr
        abs_diffs = np.abs(fpr - fnr)
        min_index = np.argmin(abs_diffs)
        eer = (fpr[min_index] + fnr[min_index]) / 2
    return eer, roc_auc

def extract_bank(model, train_loader, device, memory_bank_path):
    """
    提取训练集 Real 样本构建 Memory Bank
    """
    model.eval()
    print("Building k-NN Memory Bank from Training Data...")
    memory_bank = []
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Extracting Train Feats"):
            # 只需要 Real 样本 (Label=1)
            audio = batch["audio"][batch["label"]==1].to(device)
            if len(audio) == 0: continue
            
            if len(audio.shape) == 3: audio = audio[:, 0, :]
            
            res = model.model(audio)
            # 归一化很重要
            feat = F.normalize(res["final_feat"], p=2, dim=1)
            memory_bank.append(feat.cpu().numpy())
            
    memory_bank = np.concatenate(memory_bank, axis=0)
    print(f"Memory Bank Size: {memory_bank.shape}") 
    # 保存为 numpy 格式更通用，或者 torch
    torch.save(memory_bank, memory_bank_path)
    return memory_bank

def visualize_knn_tsne(memory_bank, test_feats, test_labels, save_path="tsne_knn_bank.png"):
    """
    绘制 t-SNE: Memory Bank (Train Real) vs Test Real vs Test Fake
    """
    print(f"Computing t-SNE for Visualization (saving to {save_path})...")
    
    # ---------------------------------------
    # 1. 数据采样 (为了速度和图表清晰)
    # ---------------------------------------
    n_samples_bank = 1000 # Memory Bank 采样数
    n_samples_test = 500  # 测试集 Real/Fake 各采样数
    
    # (A) Memory Bank 采样
    if len(memory_bank) > n_samples_bank:
        idx = np.random.choice(len(memory_bank), n_samples_bank, replace=False)
        bank_sampled = memory_bank[idx]
    else:
        bank_sampled = memory_bank

    # (B) Test Real 采样
    real_idx = np.where(test_labels == 1)[0]
    if len(real_idx) > n_samples_test:
        real_idx = np.random.choice(real_idx, n_samples_test, replace=False)
    test_real_sampled = test_feats[real_idx]

    # (C) Test Fake 采样
    fake_idx = np.where(test_labels == 0)[0]
    if len(fake_idx) > n_samples_test:
        fake_idx = np.random.choice(fake_idx, n_samples_test, replace=False)
    test_fake_sampled = test_feats[fake_idx]

    # ---------------------------------------
    # 2. 合并数据 & t-SNE
    # ---------------------------------------
    # 结构: [Memory_Bank, Test_Real, Test_Fake]
    X_all = np.concatenate([bank_sampled, test_real_sampled, test_fake_sampled], axis=0)
    
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(X_all)
    
    # 拆分坐标
    idx = 0
    p_bank = X_embedded[idx : idx + len(bank_sampled)]; idx += len(bank_sampled)
    p_real = X_embedded[idx : idx + len(test_real_sampled)]; idx += len(test_real_sampled)
    p_fake = X_embedded[idx : idx + len(test_fake_sampled)]; idx += len(test_fake_sampled)
    
    # ---------------------------------------
    # 3. 绘图
    # ---------------------------------------
    plt.figure(figsize=(12, 10))
    
    # 绘制 Memory Bank (背景，蓝色/灰色) 
    plt.scatter(
        p_bank[:, 0], p_bank[:, 1], 
        c='blue', label='Memory Bank (Train Real)', alpha=0.15, s=20, marker='.'
    )
    
    # 绘制 Test Real (绿色)
    plt.scatter(
        p_real[:, 0], p_real[:, 1], 
        c='green', label='Test Real', alpha=0.6, s=20, marker='o', edgecolors='none'
    )
    
    # 绘制 Test Fake (红色)
    plt.scatter(
        p_fake[:, 0], p_fake[:, 1], 
        c='red', label='Test Fake', alpha=0.5, s=20, marker='x'
    )
    
    plt.title("t-SNE: Memory Bank Support vs. Test Data")
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    # plt.show()

def evaluate_knn(model, memory_bank, test_loader, device, k=5, visualize=False, save_name="knn_vis.png"):
    print(f"Evaluating Test Data with k-NN (k={k})...")
    
    all_scores = []
    all_labels = []
    
    # 用于可视化的容器
    vis_feats = []
    vis_labels = []
    
    # 1. 拟合 k-NN
    knn = NearestNeighbors(n_neighbors=k, metric="cosine", n_jobs=-1)
    knn.fit(memory_bank)
    
    model.eval()
    with torch.no_grad():
        idx = 0
        for batch in tqdm(test_loader, desc="Testing"):
            idx += 1
            audio = batch["audio"].to(device)
            label = batch["label"]
            
            if len(audio.shape) == 3: audio = audio[:, 0, :]
            
            res = model.model(audio)
            # 归一化
            feat_test = F.normalize(res["final_feat"], p=2, dim=1) 
            
            # 查询距离
            feat_np = feat_test.cpu().numpy()
            dists, _ = knn.kneighbors(feat_np)
            
            # Score = 平均距离
            knn_score = dists.mean(axis=1)
            
            all_scores.extend(knn_score)
            all_labels.extend(label.cpu().numpy())
            
            vis_feats.append(feat_np)
            vis_labels.append(label.cpu().numpy())
            
            # 这里控制评估数量，正式跑可以去掉或设大点
            if idx == 50: break

    # 2. 计算指标
    y_true_anomaly = 1 - np.array(all_labels) # Real=0, Fake=1
    scores = np.array(all_scores)
    
    eer, auc_score = compute_eer(y_true_anomaly, scores)
    
    print(f"\n[k-NN Result (k={k})]")
    print(f"  > AUC : {auc_score*100:.2f}%")
    print(f"  > EER : {eer*100:.2f}%")
    
    # 3. 调用可视化
    # if visualize and len(vis_feats) > 0:
    #     vis_feats = np.concatenate(vis_feats, axis=0)
    #     vis_labels = np.concatenate(vis_labels, axis=0)
        
    #     visualize_knn_tsne(
    #         memory_bank=memory_bank,
    #         test_feats=vis_feats,
    #         test_labels=vis_labels,
    #         save_path=save_name
    #     )
    
    return eer, auc_score

if __name__ == "__main__":   
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="GMM")
    parser.add_argument("--gpu", type=int, nargs="+", default=0)
    parser.add_argument("-v", "--version", type=int, default=None)
    parser.add_argument("--test_noise", type=int, default=0)
    parser.add_argument("-nr","--noise_ratio",type=float,default=0.5) 
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("-ckpt", "--checkpoint", type=str, 
                        default="/home/zyz/data/test_controlled_ex/1222_XLSR_data_aug_RobustDataAugmentor(p=1.0)/MultiView/ASV2021_LA/version_0/checkpoints/best-epoch=10-val-eer=0.0979.ckpt")
    args = parser.parse_args()
    
    # 1. 配置
    cfg = get_cfg_defaults("config/experiments/%s.yaml" % args.cfg)
    cfg.DATASET.batch_size = args.batch_size
    cfg.DATASET.test_batch_size = args.batch_size
    device = torch.device(f'cuda:{args.gpu[0]}' if torch.cuda.is_available() else 'cpu')
    
    # 2. 模型
    print(f"Loading model from {args.checkpoint} ...")
    model = ALDA_OneClass_AugImmunity_wepe_Lit(proj=True)
    sd = torch.load(args.checkpoint, map_location="cpu")["state_dict"]
    if "loss_fn.centroid" in sd: sd["centroid"] = sd.pop("loss_fn.centroid")
    model.load_state_dict(sd, strict=False)
    model = model.to(device)
    model.eval()

    # 3. 数据
    ds, dl = make_data(cfg.DATASET, args=args)
    # 获取训练集用于构建 Bank
    train_ds, train_dl = get_true(["ASV2021_inner","ASV2021_LA"],"val") 
    test_dataloaders = dl.test if isinstance(dl.test, list) else [dl.test]
    
    memory_bank_path = "memory_bank_RobustDataAugmentor(p=1.0)_epoch=10-val-eer=0.0979.pt"
    memory_bank = extract_bank(model, train_dl, device, memory_bank_path)
    print(f"Loading Memory Bank from {memory_bank_path}...")
    memory_bank = torch.load(memory_bank_path,weights_only=False)
    if isinstance(memory_bank, torch.Tensor):
        memory_bank = memory_bank.numpy()

    save_dir = "tsne_RobustDataAugmentor(p=1.0)"
    os.makedirs(save_dir,exist_ok=True)
    # 5. 评估与可视化
    for i, test_loader in enumerate(test_dataloaders):
        print(f"\nProcessing Test Set {i}...")
        save_name = save_dir+f"/tsne_knn_set{i}.png"
        
        # 开启可视化 visualize=True
        evaluate_knn(
            model, 
            memory_bank, 
            test_loader, 
            device, 
            visualize=True, 
            save_name=save_name
        )