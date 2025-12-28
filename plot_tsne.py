import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from tqdm import tqdm
import seaborn as sns

from Oneclass_XLSR_lit import ALDA_OneClass_Lit
from Wepe_wavlm import WePeDetector
from config.config import get_cfg_defaults
from data.make_dataset import make_data

def plot_perturbation_tsne(original_model, perturbed_model, dataloader, 
                           device='cuda', max_samples=2000, 
                           save_path='perturbation_tsne.png'):
    """
    绘制扰动前后 Real/Fake 特征分布的 T-SNE 对比图
    
    Args:
        original_model: 原始模型 (Clean)
        perturbed_model: 扰动后的模型 (Noisy) 或 开启了 Dropout 的模型
        dataloader: 数据加载器
        device: 设备
        max_samples: 限制采样的最大数量，防止 T-SNE 跑太慢
        save_path: 图片保存路径
    """
    original_model.eval()
    original_model.to(device)
    
    # 如果 perturbed_model 是同一个对象（例如 MC Dropout），需要手动切换状态
    # 这里假设传入的是两个不同的模型对象（如 WePeDetector 生成的）
    perturbed_model.eval() 
    perturbed_model.to(device)

    # 1. 提取质心 (如果存在)
    centroid = None
    centroid = original_model.centroid.detach().cpu().numpy()

    # 2. 收集数据
    print(f"正在提取特征用于 T-SNE (Max samples: {max_samples})...")
    
    feats_clean = []
    feats_noisy = []
    labels_list = []
    
    sample_count = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader):
            if isinstance(batch, dict):
                audio = batch['audio']
                labels = batch['label']
            else:
                audio = torch.stack([item['audio'] for item in batch])
                labels = torch.tensor([item['label'] for item in batch])
            
            audio = audio.to(device)
            
            # 提取 Clean 特征
            # 假设模型输出 dict, 取 'final_feat' 并做 mean pooling
            out_clean = original_model.model(audio)
            f_clean = out_clean['final_feat'].mean(1) if isinstance(out_clean, dict) else out_clean.mean(1)
            
            # 提取 Noisy 特征
            out_noisy = perturbed_model.model(audio)
            f_noisy = out_noisy['final_feat'].mean(1) if isinstance(out_noisy, dict) else out_noisy.mean(1)
            
            feats_clean.append(f_clean.cpu())
            feats_noisy.append(f_noisy.cpu())
            labels_list.append(labels.cpu())
            
            sample_count += audio.size(0)
            if sample_count >= max_samples:
                break
    
    # 堆叠数据
    feats_clean = torch.cat(feats_clean, dim=0)[:max_samples].numpy()
    feats_noisy = torch.cat(feats_noisy, dim=0)[:max_samples].numpy()
    labels = torch.cat(labels_list, dim=0)[:max_samples].numpy()
    
    # 3. 准备 T-SNE 输入
    # 关键：必须把 Clean, Noisy 和 Centroid 放在一起做 t-SNE，才能保证在同一个空间坐标系下对比
    
    # 数据结构: [Clean Samples, Noisy Samples, Centroid(optional)]
    if centroid is not None:
        # Centroid 需要 reshape 成 (1, D)
        if len(centroid.shape) == 1:
            centroid = centroid.reshape(1, -1)
        combined_data = np.concatenate([feats_clean, centroid], axis=0)
    else:
        combined_data = np.concatenate([feats_clean, feats_noisy], axis=0)
        
    print(f"开始运行 T-SNE (Total points: {combined_data.shape[0]})...")
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    embedded_data = tsne.fit_transform(combined_data)
    
    # 4. 拆分数据用于画图
    n_samples = feats_clean.shape[0]
    
    emb_clean = embedded_data[:n_samples]
    emb_noisy = embedded_data[n_samples:2*n_samples]
    emb_centroid = embedded_data[-1] if centroid is not None else None
    
    # 5. 绘图
    plt.figure(figsize=(12, 10))
    sns.set_style("whitegrid")
    
    # 定义颜色和样式
    # Label 1: Real, Label 0: Fake
    
    # 筛选索引
    idx_real = (labels == 1)
    idx_fake = (labels == 0)
    
    # A. 画 Real 样本 (Clean vs Noisy)
    # 用箭头连接同一个样本的 Clean 和 Noisy 点（可选，如果点太密可以注释掉）
    # for i in np.where(idx_real)[0][:100]: # 只画前100个样本的连线，防止太乱
    #     plt.arrow(emb_clean[i,0], emb_clean[i,1], 
    #               emb_noisy[i,0]-emb_clean[i,0], emb_noisy[i,1]-emb_clean[i,1], 
    #               color='blue', alpha=0.1, width=0.002)

    plt.scatter(emb_clean[idx_real, 0], emb_clean[idx_real, 1], 
                c='dodgerblue', label='Real (Clean)', alpha=0.6, s=30, edgecolors='w')
    # plt.scatter(emb_noisy[idx_real, 0], emb_noisy[idx_real, 1], 
    #             c='cyan', label='Real (Perturbed)', alpha=0.6, marker='x', s=30)

    # B. 画 Fake 样本 (Clean vs Noisy)
    plt.scatter(emb_clean[idx_fake, 0], emb_clean[idx_fake, 1], 
                c='crimson', label='Fake (Clean)', alpha=0.6, s=30, edgecolors='w')
    # plt.scatter(emb_noisy[idx_fake, 0], emb_noisy[idx_fake, 1], 
    #             c='orange', label='Fake (Perturbed)', alpha=0.6, marker='x', s=30)
    
    # C. 画质心
    if emb_centroid is not None:
        plt.scatter(emb_centroid[0], emb_centroid[1], 
                    c='black', label='Train Centroid', s=300, marker='*', edgecolors='gold', linewidth=2)

    plt.legend(fontsize=12, loc='best')
    plt.title("T-SNE: Feature Distribution Shift under Perturbation", fontsize=15)
    plt.tight_layout()
    
    plt.savefig(save_path, dpi=300)
    print(f"T-SNE 图已保存至: {save_path}")
    plt.show()

# ================= 使用示例 =================
# 假设 model 是你的原始模型
# detector.perturbed_models[0] 是你生成的一个扰动模型
# dl.test[i] 是你的测试集 DataLoader
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
    model = ALDA_OneClass_Lit()

    sd = torch.load("/home/zyz/data/test_controlled_ex/1219_XLSR_oneclass_val/MultiView/ASV2021_LA/version_0/checkpoints/best-epoch=14-val-auc=0.9435.ckpt", map_location="cpu")["state_dict"]
    model.load_state_dict(sd,strict=False)
    model = model.to(device=device)
    center = model.centroid
    sd = torch.load("/home/zyz/data/test_controlled_ex/1217_ALDA_oneclass/MultiView/ASV2021_LA/version_0/checkpoints/best-epoch=6-val-auc=0.9630.ckpt", map_location="cpu")["state_dict"]
    model.load_state_dict(sd,strict=False)
    model.centroid = center
    ds, dl = make_data(cfg.DATASET, args=args)
    ds_name = args.cfg.split("/")[1]
    # 创建检测器 - 预先生成扰动模型
    start_layer = 0
    end_layer = 6
    detector = WePeDetector(
        model, 
        noise_type="uniform",
        device=device,
        num_perturbations=1,      # 预生成10个扰动模型
        layers_to_perturb=[i for i in range(start_layer,end_layer)],  # 扰动前两层
        noise_ratio=args.noise_ratio
    )
    for i in range(len(dl.test)):
        plot_perturbation_tsne(model, detector.perturbed_models[0], dl.test[i], 
                            device=device, max_samples=1000, 
                            save_path=f'analysis_tsne_[{i}].png')