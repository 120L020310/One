import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os

# 配置 seaborn 样式，让图表更美观
sns.set(style="whitegrid")

def load_memory_bank(path):
    """加载 Memory Bank 文件 (.pt)"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到文件: {path}")
    
    print(f"正在加载 Memory Bank: {path} ...")
    memory_bank = torch.load(path, map_location='cpu')
    
    # 转换为 numpy
    if isinstance(memory_bank, torch.Tensor):
        memory_bank = memory_bank.detach().numpy()
        
    print(f"加载成功。特征形状: {memory_bank.shape}")
    return memory_bank

def plot_2d_projection(features, method='tsne', sample_limit=5000,fig_name=None):
    """
    使用 t-SNE 或 PCA 将特征投影到 2D 平面并画图。
    为了速度，如果特征数量巨大，会进行采样。
    """
    N, D = features.shape
    
    # 如果数据太多，随机采样一部分进行可视化（t-SNE 跑太慢）
    if N > sample_limit:
        print(f"数据点过多 ({N})，随机采样 {sample_limit} 个用于 {method} 可视化...")
        indices = np.random.choice(N, sample_limit, replace=False)
        features_subset = features[indices]
    else:
        features_subset = features

    print(f"正在进行 {method.upper()} 降维...")
    
    if method.lower() == 'tsne':
        # Perplexity 通常设为 30-50
        reducer = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, init='pca', learning_rate='auto')
    else:
        reducer = PCA(n_components=2)
        
    projections = reducer.fit_transform(features_subset)
    
    # 绘图
    plt.figure(figsize=(10, 8))
    # 使用 scatter plot，alpha 设置透明度可以看出密度
    plt.scatter(projections[:, 0], projections[:, 1], alpha=0.6, s=10, c='#3498db', edgecolors='none')
    plt.title(f"Feature Distribution ({method.upper()}) - Shape: {features_subset.shape}")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    
    # 去除边框
    sns.despine()
    plt.tight_layout()
    plt.savefig(f"/home/zyz/work2_dir/Wepe_DFM_11_18/tsne_img/{fig_name}_{method}.png", dpi=300)
    print(f"已保存图像: {fig_name}_{method}.png")
    plt.show()

def plot_norm_distribution(features,fig_name):
    """绘制特征向量的 L2 范数分布"""
    print("计算特征范数分布...")
    # 计算每个向量的 L2 范数 (模长)
    norms = np.linalg.norm(features, axis=1)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(norms, kde=True, bins=50, color='#e74c3c')
    plt.title("Distribution of Feature L2 Norms")
    plt.xlabel("L2 Norm Value")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"/home/zyz/work2_dir/Wepe_DFM_11_18/tsne_img/{fig_name}_norm.png", dpi=300)
    print(f"已保存图像: /home/zyz/work2_dir/Wepe_DFM_11_18/tsne_img/{fig_name}_norm.png")
    plt.show()

def plot_value_histogram(features,fig_name):
    """绘制所有特征值的直方图 (检查是否存在 Dead Neurons 或 异常值)"""
    print("计算特征数值分布...")
    # 展平所有数值
    values = features.flatten()
    
    # 如果数据量太大，随机采样一部分画图
    if len(values) > 100000:
        values = np.random.choice(values, 100000, replace=False)
        
    plt.figure(figsize=(10, 6))
    sns.histplot(values, kde=True, bins=100, color='#2ecc71')
    plt.title("Histogram of Feature Activation Values (Subsampled)")
    plt.xlabel("Activation Value")
    plt.yscale('log') # 使用对数坐标，因为 0 附近的值可能非常多
    plt.ylabel("Count (Log Scale)")
    plt.tight_layout()
    plt.savefig(f"/home/zyz/work2_dir/Wepe_DFM_11_18/tsne_img/{fig_name}_values.png", dpi=300)
    print(f"已保存图像: {fig_name}_values.png")
    plt.show()

if __name__ == "__main__":
    # ================= 配置 =================
    MEMORY_BANK_dir_PATH = "/home/zyz/data/work2/memory_bank_librispeech"  # 你的文件路径
    # =======================================
    for dirpath, dirnames, filenames in os.walk(MEMORY_BANK_dir_PATH):
        for filename in filenames:
            try:
                # 1. 加载数据
                MEMORY_BANK_PATH = os.path.join(dirpath, filename)
                features = load_memory_bank(MEMORY_BANK_PATH)
                
                # 2. 绘制 t-SNE (推荐，最能反映局部结构)
                # 注意：如果数据量特别大，t-SNE 会很慢，脚本里自动做了采样
                plot_2d_projection(features, method='tsne',fig_name=filename.split(".")[0])
                
                # # 3. 绘制 PCA (反映全局方差)
                # # plot_2d_projection(features, method='pca')
                
                # # 4. 绘制范数分布 (检查特征强度)
                # plot_norm_distribution(features,fig_name=filename.split(".")[0])
                
                # # 5. 绘制数值分v布 (检查数值范围)
                # plot_value_histogram(features,fig_name=filename.split(".")[0])
                
                print("\n所有可视化完成！请查看生成的 .png 图片。")
                
            except Exception as e:
                print(f"发生错误: {e}")