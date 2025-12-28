import copy
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve
import torch
import torch.nn.functional as F

def cosine_based_sim_scores(sample_features):
    """基于余弦相似度的不确定性计算"""
    # 计算相似度矩阵 [num_perturbations, num_perturbations]
    similarity = torch.matmul(
        sample_features[0], 
        sample_features[1]
    )
    
    # 计算不确定性：相似度越低，不确定性越高
    uncertainty = 2 - 2 * similarity
    
    return uncertainty, similarity

def cosine_based_uncertainty(sample_features):
    """基于余弦相似度的不确定性计算"""
    # 计算相似度矩阵 [num_perturbations, num_perturbations]
    similarity_matrix = F.cosine_similarity(
        sample_features.unsqueeze(1), 
        sample_features.unsqueeze(0), 
        dim=2
    )
    
    # 取上三角部分（不包括对角线）
    mask = torch.triu(torch.ones_like(similarity_matrix), diagonal=1).bool()
    upper_triangle_similarities = similarity_matrix[mask]
    
    # 计算平均相似度
    avg_similarity = upper_triangle_similarities.mean().item()
    
    # 计算不确定性：相似度越低，不确定性越高
    uncertainty = 2 - 2 * avg_similarity
    
    return uncertainty, avg_similarity

def euclidean_based_uncertainty(sample_features):
    """基于欧氏距离的不确定性计算"""
    num_perturbations = sample_features.size(0)
    
    # 计算欧氏距离矩阵 [num_perturbations, num_perturbations]
    distance_matrix = torch.cdist(sample_features, sample_features, p=2)
    
    # 取上三角部分（不包括对角线）
    mask = torch.triu(torch.ones_like(distance_matrix), diagonal=1).bool()
    upper_triangle_distances = distance_matrix[mask]
    
    # 计算平均距离
    avg_distance = upper_triangle_distances.mean().item()
    
    # 归一化距离到[0,1]范围，然后转换为不确定性
    # 距离越大，不确定性越高
    max_distance = torch.norm(sample_features, p=2, dim=1).max().item() * 2  # 估计最大可能距离
    normalized_distance = min(avg_distance / max_distance, 1.0) if max_distance > 0 else 0
    
    uncertainty = normalized_distance
    
    return uncertainty, avg_distance

def manhattan_based_uncertainty(sample_features):
    """基于曼哈顿距离的不确定性计算"""
    num_perturbations = sample_features.size(0)
    
    # 计算曼哈顿距离矩阵 [num_perturbations, num_perturbations]
    distance_matrix = torch.cdist(sample_features, sample_features, p=1)
    
    # 取上三角部分（不包括对角线）
    mask = torch.triu(torch.ones_like(distance_matrix), diagonal=1).bool()
    upper_triangle_distances = distance_matrix[mask]
    
    # 计算平均距离
    avg_distance = upper_triangle_distances.mean().item()
    
    # 归一化距离到[0,1]范围
    max_distance = torch.norm(sample_features, p=1, dim=1).max().item() * 2
    normalized_distance = min(avg_distance / max_distance, 1.0) if max_distance > 0 else 0
    
    uncertainty = normalized_distance
    
    return uncertainty, avg_distance

def mahalanobis_based_uncertainty(sample_features):
    """基于马氏距离的不确定性计算"""
    num_perturbations, feature_dim = sample_features.shape
    
    if num_perturbations <= feature_dim:
        # 样本数不足，回退到欧氏距离
        return euclidean_based_uncertainty(sample_features)
    
    try:
        # 计算协方差矩阵的逆
        cov_matrix = torch.cov(sample_features.T)
        inv_cov_matrix = torch.linalg.pinv(cov_matrix)
        
        # 计算均值向量
        mean_vector = sample_features.mean(dim=0)
        
        # 计算每个样本到均值的马氏距离
        deviations = sample_features - mean_vector.unsqueeze(0)
        mahalanobis_distances = torch.sqrt(
            torch.sum(deviations @ inv_cov_matrix * deviations, dim=1)
        )
        
        # 计算平均马氏距离
        avg_distance = mahalanobis_distances.mean().item()
        
        # 马氏距离的归一化较复杂，这里使用经验值
        # 通常马氏距离的平方服从卡方分布，但这里简化处理
        normalized_distance = min(avg_distance / 10.0, 1.0)  # 经验归一化
        
        uncertainty = normalized_distance
        return uncertainty, avg_distance
        
    except Exception as e:
        print(f"马氏距离计算失败: {e}，回退到欧氏距离")
        return euclidean_based_uncertainty(sample_features)

def chebyshev_based_uncertainty(sample_features):
    """基于切比雪夫距离的不确定性计算"""
    num_perturbations = sample_features.size(0)
    
    # 计算切比雪夫距离矩阵 [num_perturbations, num_perturbations]
    distance_matrix = torch.cdist(sample_features, sample_features, p=float('inf'))
    
    # 取上三角部分（不包括对角线）
    mask = torch.triu(torch.ones_like(distance_matrix), diagonal=1).bool()
    upper_triangle_distances = distance_matrix[mask]
    
    # 计算平均距离
    avg_distance = upper_triangle_distances.mean().item()
    
    # 归一化距离到[0,1]范围
    max_possible = sample_features.abs().max().item() * 2
    normalized_distance = min(avg_distance / max_possible, 1.0) if max_possible > 0 else 0
    
    uncertainty = normalized_distance
    
    return uncertainty, avg_distance

def correlation_based_uncertainty(sample_features):
    """基于相关系数的不确定性计算"""
    num_perturbations = sample_features.size(0)
    
    # 计算相关系数矩阵
    correlation_matrix = torch.corrcoef(sample_features)
    
    # 取上三角部分（不包括对角线）
    mask = torch.triu(torch.ones_like(correlation_matrix), diagonal=1).bool()
    upper_triangle_correlations = correlation_matrix[mask]
    
    # 计算平均相关系数
    avg_correlation = upper_triangle_correlations.mean().item()
    
    # 相关系数范围[-1,1]，映射到不确定性[0,2]
    # 相关性越低，不确定性越高
    uncertainty = 1 - avg_correlation  # 范围[0,2]
    
    return uncertainty, avg_correlation

def should_perturb(param_name, layers_to_perturb):
    """
    判断参数是否属于需要扰动的层
    param_name 示例: "xlsr.wav2vec2.encoder.layers.0.attention.k_proj.weight"
    """
    # 1. 检查是否在指定的 Transformer Layers 中
    if layers_to_perturb is not None:
        for layer_idx in layers_to_perturb:
            # 关键修改：加上 "." (点)，确保精确匹配
            # 防止输入 1 时匹配到 10, 11 等 (虽然wav2vec2命名规范通常没事，但加上更安全)
            target_str = f"encoder.layers.{layer_idx}."
            if target_str in param_name:
                return True
    
    # 2. (可选) 是否扰动底层的 CNN 特征提取器和位置编码？
    # 如果你希望扰动甚至比 Layer 0 更“底层”的特征，取消下面的注释
    # if "feature_extractor" in param_name or "pos_conv_embed" in param_name:
    #     return True

    return False

def is_weight_parameter(name, param):
    """判断是否为权重参数"""
    # 1. 维度检查: 卷积核(3D/4D) 或 Linear权重(2D)
    if param.dim() < 2:
        return False
    
    # 2. 名字过滤: 跳过 Norm 和 Bias
    # Wav2Vec2 中 LayerNorm 的参数名通常包含 'layer_norm'
    if 'layer_norm' in name or 'norm' in name:
        return False
        
    if 'bias' in name:
        return False
        
    return True

def confusion_matrix(labels, predictions):
    """计算混淆矩阵"""
    tp = np.sum((predictions == 1) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))
    return tn, fp, fn, tp

def find_optimal_threshold_by_eer(results):
    """
    专门基于EER寻找最优阈值
    """
    uncertainties = results['uncertainties']
    labels = results['labels']
    
    # 计算精确的EER
    fpr, tpr, thresholds = roc_curve(labels, uncertainties)
    fnr = 1 - tpr
    
    # 找到EER点 (FAR = FRR)
    eer_index = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_index] + fnr[eer_index]) / 2
    eer_threshold = thresholds[eer_index]
    
    print(f"\n=== 基于EER的阈值选择 ===")
    print(f"EER: {eer:.4f}")
    print(f"EER阈值: {eer_threshold:.6f}")
    
    # 计算在该阈值下的性能
    predictions = (uncertainties > eer_threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions)
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    print(f"在EER阈值下的性能:")
    print(f"  准确率: {accuracy:.4f}")
    print(f"  精确率: {precision:.4f}")
    print(f"  召回率: {recall:.4f}")
    print(f"  F1分数: {f1:.4f}")
    print(f"  混淆矩阵: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    
    return eer_threshold, eer, accuracy, f1

def compute_eer(labels, scores,fanzhuan=True):
    """
    计算等错误率 (EER)
    """
    # 计算FAR和FRR
    if fanzhuan:
        labels = 1-labels
    fpr, tpr, thresholds = roc_curve(labels, scores)
    fnr = 1 - tpr
    
    # 找到EER点 (FAR = FRR)
    eer_index = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_index] + fnr[eer_index]) / 2
    eer_threshold = thresholds[eer_index]
    
    return eer, eer_threshold

def compute_metrics_improved(uncertainties, labels, thresholds=None):
    """
    改进的阈值选择方法 - 基于您的修改补充完整逻辑
    """
    # 您的现有代码：阈值生成和循环
    if thresholds is None:
        min_val = 0.8
        max_val = 1
        low_thresholds = np.linspace(min_val, 0.9, 20)
        mid_thresholds = np.linspace(0.9, 0.99, 20)
        high_thresholds = np.linspace(0.99, max_val, 20)
        thresholds = np.concatenate([low_thresholds, mid_thresholds, high_thresholds])
    
    best_f1 = 0
    best_gmean = 0
    best_combined = 0
    best_threshold_f1 = 0
    best_threshold_gmean = 0
    best_threshold_combined = 0
    
    all_metrics = []
    
    for threshold in thresholds:
        predictions = (uncertainties > threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, predictions)
        
        # 计算基础指标
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        gmean = np.sqrt(recall * specificity)
        
        # 新增：综合评分（可根据需求调整权重）
        # 权重说明：召回率重要（抓伪造），但也要保护真实样本（特异度）
        combined_score = 0.4 * recall + 0.4 * specificity + 0.2 * f1
        
        metrics = {
            'threshold': threshold, 'f1': f1, 'gmean': gmean,
            'precision': precision, 'recall': recall, 'specificity': specificity,
            'combined_score': combined_score,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        }
        all_metrics.append(metrics)
        
        # 更新最佳阈值
        if f1 > best_f1:
            best_f1 = f1
            best_threshold_f1 = threshold
            
        if gmean > best_gmean:
            best_gmean = gmean
            best_threshold_gmean = threshold
            
        if combined_score > best_combined:
            best_combined = combined_score
            best_threshold_combined = threshold
    
    # 关键修改：智能阈值选择策略
    # 计算精确EER（在循环外部调用一次）
    eer, eer_threshold = compute_eer(labels, uncertainties)
    
    # 方案A：优先考虑综合评分
    best_threshold = best_threshold_combined
    selection_method = "Combined_Score"
    
    # 方案B：如果EER阈值在合理范围内，优先使用EER
    real_scores = uncertainties[labels == 1]
    fake_scores = uncertainties[labels == 0]
    real_q3 = np.percentile(real_scores, 75)  # ~0.0146
    fake_q1 = np.percentile(fake_scores, 25)  # ~0.0162
    
    # 检查EER阈值是否在统计合理的范围内
    if real_q3 <= eer_threshold <= fake_q1:
        best_threshold = eer_threshold
        selection_method = "EER_Statistical"
        print(f"使用EER阈值(统计合理): {eer_threshold:.4f}")
    else:
        print(f"EER阈值{eer_threshold:.4f}不在统计合理范围[{real_q3:.4f}, {fake_q1:.4f}]内")
        print(f"使用综合评分阈值: {best_threshold:.4f}")
    
    return {
        'uncertainties': uncertainties,
        "labels": labels,
        'best_threshold': best_threshold,
        'best_f1': best_f1,
        'best_gmean': best_gmean,
        'best_combined': best_combined,
        'best_threshold_f1': best_threshold_f1,
        'best_threshold_gmean': best_threshold_gmean,
        'best_threshold_combined': best_threshold_combined,
        'eer': eer,
        'eer_threshold': eer_threshold,
        'real_q3': real_q3,
        'fake_q1': fake_q1,
        'all_metrics': all_metrics,
        'selection_method': selection_method
    }

# 其他函数保持不变...
import torch.distributions as dist # 用于拉普拉斯噪声

def perturb_weights(device, 
                    model, 
                    noise_type='gaussian', 
                    noise_ratio=0.1, 
                    layers_to_perturb=None):
    """
    对模型权重添加指定类型的扰动。
    
    根据 WePe 论文 (arXiv:2412.05897) ，
    支持 'gaussian', 'uniform', 'laplace' 噪声。

    参数:
    - device: torch.device
    - model: 原始的 PyTorch 模型
    - noise_type: 噪声类型 ('gaussian', 'uniform', 'laplace')
    - noise_ratio: 噪声比例 (论文中设为 0.1 )，用于计算噪声的标准差
    - layers_to_perturb: 需要扰动的层列表 (例如 DINOv2 ViT-L/14 的前 19 个块 [cite: 520])
    """
    
    # 1. 深度拷贝模型
    perturbed_model = copy.deepcopy(model)
    perturbed_model.to(device)
    perturbed_model.eval()
    
    with torch.no_grad():
        for name, param in perturbed_model.named_parameters():
            
            # 2. 检查是否为需要扰动的层
            if layers_to_perturb is not None:
                if not should_perturb(name, layers_to_perturb):
                    continue
            
            # 3. 检查是否为权重参数 (非 bias/norm)
            if is_weight_parameter(name, param):
                
                # 4. 计算目标噪声标准差
                std_dev = noise_ratio * torch.mean(torch.abs(param))
                # 5. 根据类型生成噪声
                if noise_type == 'gaussian':
                    noise = torch.randn_like(param) * std_dev
                elif noise_type == 'uniform':
                    # 均匀分布 Uniform(-a, a) 的标准差 std = a / sqrt(3)
                    a = std_dev * (3**0.5)
                    noise = (torch.rand_like(param) * 2 - 1) * a # 范围 [-a, a]
                elif noise_type == 'laplace':
                    # 拉普拉斯分布 Laplace(0, b) 的标准差 std = b * sqrt(2)
                    scale = std_dev / (2**0.5)
                    laplace_dist = dist.Laplace(0.0, scale)
                    noise = laplace_dist.sample(param.size()).to(param.device)
                else:
                    raise ValueError(f"不支持的噪声类型: {noise_type}. "
                                     f"请从 'gaussian', 'uniform', 'laplace' 中选择。")
                # 6. 添加噪声
                param.add_(noise)
                
    return perturbed_model
def add_weight_noise(model, noise_type='uniform', noise_ratio=0.01, layers_to_perturb=None):
    """
    [In-Place Version] 对模型权重添加扰动，并返回噪声字典以便恢复。
    完全复刻 WePe 论文逻辑。
    """
    noise_dict = {}
    device = next(model.parameters()).device
    
    # 遍历所有参数
    for name, param in model.named_parameters():
        # 0. 基础过滤：必须是参与梯度的参数
        if not param.requires_grad:
            continue
            
        # 1. 检查是否在指定层中
        if layers_to_perturb is not None:
            if not should_perturb(name, layers_to_perturb):
                continue
        
        # 2. 检查是否为权重参数 (排除 bias/norm)
        if is_weight_parameter(name, param):
            
            # 3. 计算目标噪声标准差 (Relative STD)
            # std = ratio * mean(|w|)
            std_dev = noise_ratio * torch.mean(torch.abs(param))
            
            # 防止 std_dev 为 0 (例如参数全为0的情况)
            std_dev = std_dev + 1e-9
            
            # 4. 根据类型生成噪声
            if noise_type == 'gaussian':
                noise = torch.randn_like(param) * std_dev
                
            elif noise_type == 'uniform':
                # Uniform(-a, a) -> std = a / sqrt(3) -> a = std * sqrt(3)
                a = std_dev * (3**0.5)
                noise = (torch.rand_like(param) * 2 - 1) * a
                
            elif noise_type == 'laplace':
                # Laplace(0, b) -> std = b * sqrt(2) -> b = std / sqrt(2)
                scale = std_dev / (2**0.5)
                # 使用 PyTorch 的分布采样
                laplace_dist = dist.Laplace(0.0, scale)
                noise = laplace_dist.sample(param.size()).to(device)
            else:
                raise ValueError(f"Unsupported noise type: {noise_type}")
            
            # 5. [核心] 原地添加噪声
            param.data.add_(noise)
            
            # 6. 保存噪声以便恢复
            noise_dict[name] = noise
            
    return noise_dict

from contextlib import contextmanager
import torch.nn as nn
@contextmanager
def temporary_dropout_context(model, force_rate=None):
    """
    上下文管理器：临时强制开启 Dropout 并冻结 BN，
    退出上下文时自动恢复模型原始状态。
    
    用于对抗训练 (Consistency Training) 中生成 noisy view。
    """
    # 1. 备份原始状态 (Backup)
    # 我们需要记录每个受影响模块原本是 train 还是 eval，以及原本的 p 值
    backup_states = {}
    
    for name, m in model.named_modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            backup_states[name] = {'mode': m.training, 'p': m.p}
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            # 注意：有时也需要冻结 LayerNorm，视情况而定，这里主要关注 BN
            backup_states[name] = {'mode': m.training}

    # 2. 修改状态 (Apply Changes)
    # 必须在 self.model 原身操作，不能 copy
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train() # 强制开启 Dropout
            if force_rate is not None:
                m.p = force_rate # 强制修改概率
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            m.eval() # 强制冻结 BN 统计量 (防止噪声污染 running_mean/var)

    try:
        yield # 这里的 yield 会把控制权交还给 training_step 执行 forward
    finally:
        # 3. 恢复状态 (Restore)
        # 无论中间是否报错，都会执行这里
        for name, m in model.named_modules():
            if name in backup_states:
                state = backup_states[name]
                # 恢复 train/eval 模式
                m.train(state['mode']) 
                # 恢复 dropout 概率
                if 'p' in state:
                    m.p = state['p']

def remove_weight_noise(model, noise_dict):
    """
    [In-Place Version] 移除之前添加的噪声，恢复模型原始权重。
    """
    for name, param in model.named_parameters():
        if name in noise_dict:
            # 原地减去噪声
            param.data.sub_(noise_dict[name])
def extract_features_batch(audio, models):
    """
    批量提取特征 - 对同一音频使用多个模型
    """
    all_features = []
    
    for model in models:
        with torch.no_grad():
            x = model.model(audio)['final_feat'].mean(1)        # [bs,dim]
            features = F.normalize(x, p=2, dim=1)
            all_features.append(features)
    
    # 堆叠所有特征 [num_models, batch_size, feature_dim]
    return torch.stack(all_features, dim=0)
    
def calculate_uncertainty_3d_vectorized(perturbed_models, audio_batch):
    """
    向量化的三维不确定性计算
    """
    all_features = extract_features_batch(audio_batch, perturbed_models)
    # all_features shape: [8, 64, 149, 1024]
    
    batch_size = audio_batch.size(0)
    num_perturbations = all_features.size(0)
    time_steps = all_features.size(2)
    
    uncertainties = []
    avg_similarities = []
    
    for i in range(batch_size):
        # [num_perturbations, time_steps, feature_dim] -> [num_perturbations, time_steps, feature_dim]
        sample_features = all_features[:, i, :, :]
        
        # 方法2: 向量化计算 - 扩展维度用于广播
        # [num_perturbations, time_steps, feature_dim] -> [num_perturbations, 1, time_steps, feature_dim]
        features_a = sample_features.unsqueeze(1)
        # [num_perturbations, time_steps, feature_dim] -> [1, num_perturbations, time_steps, feature_dim]  
        features_b = sample_features.unsqueeze(0)
        
        # 计算所有配对的时间步级余弦相似度 [num_perturbations, num_perturbations, time_steps]
        similarity_matrix = F.cosine_similarity(features_a, features_b, dim=3)
        
        # 创建上三角mask [num_perturbations, num_perturbations]
        mask = torch.triu(torch.ones(num_perturbations, num_perturbations), diagonal=1).bool()
        mask = mask.unsqueeze(2).expand(-1, -1, time_steps)  # 扩展到时间步维度
        
        # 应用mask并计算每个时间步的平均相似度 [time_steps]
        masked_similarities = similarity_matrix[mask].view(-1, time_steps)
        time_step_avg_similarities = masked_similarities.mean(dim=0)
        
        # 整个样本的平均相似度（跨时间步）
        avg_similarity = time_step_avg_similarities.mean().item()
        uncertainty = 2 - 2 * avg_similarity
        
        uncertainties.append(uncertainty)
        avg_similarities.append(avg_similarity)
    
    return uncertainties, avg_similarities

# def calculate_uncertainty_batch(perturbed_models, audio_batch):
#     """
#     批量计算不确定性 - 使用预先生成的扰动模型
#     """
#     # 提取所有扰动模型的特征
#     # features_shape: [num_perturbations, batch_size, feature_dim]
#     all_features = extract_features_batch(audio_batch, perturbed_models)
    
#     batch_size = audio_batch.size(0)
#     uncertainties = []
#     avg_similarities = []
    
#     # 对批次中的每个样本分别计算
#     for i in range(batch_size):
#         # 获取该样本在所有扰动模型下的特征 [num_perturbations, feature_dim]
#         sample_features = all_features[:, i, :]
        
#         # 计算相似度矩阵 [num_perturbations, num_perturbations]
#         similarity_matrix = F.cosine_similarity(
#             sample_features.unsqueeze(1), 
#             sample_features.unsqueeze(0), 
#             dim=2
#         )
        
#         # 取上三角部分（不包括对角线）
#         mask = torch.triu(torch.ones_like(similarity_matrix), diagonal=1).bool()
#         upper_triangle_similarities = similarity_matrix[mask]
        
#         # 计算平均相似度
#         avg_similarity = upper_triangle_similarities.mean().item()
        
#         # 计算不确定性
#         uncertainty = 2 - 2 * avg_similarity
        
#         uncertainties.append(uncertainty)
#         avg_similarities.append(avg_similarity)
    
#     return uncertainties, avg_similarities
def calculate_uncertainty_batch(perturbed_models, audio_batch, method='euclidean'):  # 采用多个扰动模型计算的方法，不采用原始模型
    """
    批量计算不确定性 - 支持多种距离度量方法
    
    Args:
        audio_batch: 输入音频批次
        method: 距离度量方法，可选 'cosine', 'euclidean', 'manhattan', 'mahalanobis', 'chebyshev'
    """
    # 提取所有扰动模型的特征
    all_features = extract_features_batch(audio_batch, perturbed_models)
    
    batch_size = audio_batch.size(0)
    uncertainties = []
    avg_distances = []  # 改名为平均距离，更通用
    
    # 对批次中的每个样本分别计算
    for i in range(batch_size):
        # 获取该样本在所有扰动模型下的特征 [num_perturbations, feature_dim]
        sample_features = all_features[:, i, :]
        uncertainty, avg_distance = compute_feature_uncertainty(sample_features,method)
        
        uncertainties.append(uncertainty)
        avg_distances.append(avg_distance)
    
    return uncertainties, avg_distances

def calculate_uncertainty_batch_teacher(models, audio_batch, method='euclidean',train_centroid = None):  # 采用原始模型和单个扰动模型计算的方法
    """
    批量计算不确定性 - 支持多种距离度量方法
    
    Args:
        audio_batch: 输入音频批次
        method: 距离度量方法，可选 'cosine', 'euclidean', 'manhattan', 'mahalanobis', 'chebyshev'
    """
    # 提取所有扰动模型的特征
    all_features = extract_features_batch(audio_batch, models)
    original_feats = all_features[1]    # (Batch, Dim)
    perturbed_feats = all_features[0]   # (Batch, Dim)
    orig_norm = F.normalize(original_feats, p=2, dim=1)
    centroid_norm = F.normalize(train_centroid.to(original_feats.device), p=2, dim=0)
    
    # 计算余弦相似度 (Batch,)
    similarities = torch.matmul(orig_norm, centroid_norm)
    
    # 3. 计算 Uncertainty (Penalty): Distance between Original and Perturbed
    # 推荐使用欧氏距离作为不确定性度量
    if method == 'cosine':
        # 如果非要用余弦距离 (1 - cos)
        pert_norm = F.normalize(perturbed_feats, p=2, dim=1)
        cos_sim = (orig_norm * pert_norm).sum(dim=1)
        uncertainties = 1 - cos_sim
    else:
        # 默认 Euclidean (L2 距离)
        # dim=1 表示在特征维度求范数
        uncertainties = torch.norm(original_feats - perturbed_feats, p=2, dim=1)
        
    return uncertainties, similarities

def compute_feature_uncertainty(sample_features,method,train_centroid):
    if method == 'cosine':
            uncertainty, avg_distance = cosine_based_uncertainty(sample_features)
    elif method == 'sim':
            uncertainty, avg_distance = cosine_based_sim_scores(sample_features,train_centroid)
    elif method == 'euclidean':
        uncertainty, avg_distance = euclidean_based_uncertainty(sample_features)
    elif method == 'manhattan':
        uncertainty, avg_distance = manhattan_based_uncertainty(sample_features)
    elif method == 'mahalanobis':
        uncertainty, avg_distance = mahalanobis_based_uncertainty(sample_features)
    elif method == 'chebyshev':
        uncertainty, avg_distance = chebyshev_based_uncertainty(sample_features)
    else:
        raise ValueError(f"不支持的度量方法: {method}")
    return uncertainty, avg_distance
def analyze_separation_quality(results, classification_threshold=0.01):
    """
    分析模型在指定二分类阈值处的分离质量
    """
    uncertainties = results['uncertainties']
    labels = results['labels']
    
    # 根据您的观察，label=1是真实样本，label=0是伪造样本
    real_uncertainties = uncertainties[labels == 1]  # label=1: 真实样本
    fake_uncertainties = uncertainties[labels == 0]  # label=0: 伪造样本
    
    # 计算在二分类阈值处的分类性能
    real_below_threshold = np.sum(real_uncertainties < classification_threshold)
    real_above_threshold = np.sum(real_uncertainties >= classification_threshold)
    fake_below_threshold = np.sum(fake_uncertainties < classification_threshold)
    fake_above_threshold = np.sum(fake_uncertainties >= classification_threshold)
    
    total_real = len(real_uncertainties)
    total_fake = len(fake_uncertainties)
    
    print(f"\n在二分类阈值 {classification_threshold} 处的分离质量分析:")
    print(f"真实样本 (label=1):")
    print(f"  低于阈值: {real_below_threshold}/{total_real} ({real_below_threshold/total_real*100:.2f}%)")
    print(f"  高于阈值: {real_above_threshold}/{total_real} ({real_above_threshold/total_real*100:.2f}%)")
    
    print(f"伪造样本 (label=0):")
    print(f"  低于阈值: {fake_below_threshold}/{total_fake} ({fake_below_threshold/total_fake*100:.2f}%)")
    print(f"  高于阈值: {fake_above_threshold}/{total_fake} ({fake_above_threshold/total_fake*100:.2f}%)")
    
    # 计算准确率
    correct_predictions = real_below_threshold + fake_above_threshold
    total_samples = total_real + total_fake
    accuracy = correct_predictions / total_samples
    
    # 计算精确率、召回率和F1分数
    precision = fake_above_threshold / (fake_above_threshold + real_above_threshold) if (fake_above_threshold + real_above_threshold) > 0 else 0
    recall = fake_above_threshold / total_fake if total_fake > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # 计算当前二分类阈值下的FPR和FNR
    fpr = real_above_threshold / total_real if total_real > 0 else 0  # 假阳性率：真实样本被误判为伪造
    fnr = fake_below_threshold / total_fake if total_fake > 0 else 0  # 假阴性率：伪造样本被误判为真实
    
    print(f"\n使用二分类阈值 {classification_threshold} 的分类性能:")
    print(f"  准确率: {accuracy*100:.2f}%")
    print(f"  精确率: {precision*100:.2f}%")
    print(f"  召回率: {recall*100:.2f}%")
    print(f"  F1分数: {f1*100:.2f}%")
    print(f"  假阳性率 (FPR): {fpr*100:.2f}%")
    print(f"  假阴性率 (FNR): {fnr*100:.2f}%")
    
    # 计算EER相关信息（独立于当前二分类阈值）
    eer, eer_threshold = compute_eer(labels, uncertainties)
    print(f"\nEER相关信息（基于ROC曲线）:")
    print(f"  全局最优EER: {eer*100:.2f}%")
    print(f"  EER对应阈值: {eer_threshold:.6f}")
    
    # 计算在当前二分类阈值下的简化EER（仅用于参考）
    simplified_eer = (fpr + fnr) / 2
    print(f"  当前二分类阈值下的简化EER: {simplified_eer*100:.2f}%")
    
    return {
        'classification_threshold': classification_threshold,
        'real_below': real_below_threshold,
        'real_above': real_above_threshold,
        'fake_below': fake_below_threshold,
        'fake_above': fake_above_threshold,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fpr': fpr,
        'fnr': fnr,
        'simplified_eer': simplified_eer,
        'global_eer': eer,
        'global_eer_threshold': eer_threshold
    }

def plot_results(results, save_path=None):
    """
    绘制评估结果图表
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    uncertainties = results['uncertainties']
    labels = results['labels']
    
    # 1. 不确定性分布
    real_scores = uncertainties[labels == 1]  # label=1: 真实样本
    fake_scores = uncertainties[labels == 0]  # label=0: 伪造样本
    
    ax1.hist(real_scores, bins=50, alpha=0.7, label='Real (label=1)', density=True)
    ax1.hist(fake_scores, bins=50, alpha=0.7, label='Fake (label=0)', density=True)
    ax1.axvline(results['best_threshold'], color='red', linestyle='--', 
                label=f'Best Threshold: {results["best_threshold"]:.4f}')
    ax1.set_xlabel('Uncertainty Score')
    ax1.set_ylabel('Density')
    ax1.set_title('Uncertainty Score Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # # 2. ROC曲线
    # fpr, tpr, _ = roc_curve(labels, uncertainties)
    # ax2.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {results["auc_roc"]:.4f})')
    # ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    # ax2.set_xlabel('False Positive Rate')
    # ax2.set_ylabel('True Positive Rate')
    # ax2.set_title('ROC Curve')
    # ax2.legend()
    # ax2.grid(True, alpha=0.3)
    
    # # 3. 精确率-召回率曲线
    # precision, recall, _ = precision_recall_curve(labels, uncertainties)
    # ax3.plot(recall, precision, linewidth=2, 
    #         label=f'PR curve (AUC = {results["auc_pr"]:.4f})')
    # ax3.set_xlabel('Recall')
    # ax3.set_ylabel('Precision')
    # ax3.set_title('Precision-Recall Curve')
    # ax3.legend()
    # ax3.grid(True, alpha=0.3)
    
    # # 4. F1分数 vs 阈值
    # thresholds = [m['threshold'] for m in results['all_metrics']]
    # f1_scores = [m['f1'] for m in results['all_metrics']]
    # ax4.plot(thresholds, f1_scores, linewidth=2, label='F1 Score')
    # ax4.axvline(results['best_threshold'], color='red', linestyle='--',
    #            label=f'Best Threshold: {results["best_threshold"]:.4f}')
    # ax4.set_xlabel('Threshold')
    # ax4.set_ylabel('F1 Score')
    # ax4.set_title('F1 Score vs Threshold')
    # ax4.legend()
    # ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"图表已保存至: {save_path}")
    
    plt.show()
    
    return fig

def plot_distribution_comparison(results, save_path='distribution_comparison.png'):
    """
    绘制两个类别不确定性分布的对比图
    修正：更新标签解释
    """
    uncertainties = results['uncertainties']
    labels = results['labels']
    
    # 根据您的观察，label=1是真实样本，label=0是伪造样本
    real_uncertainties = uncertainties[labels == 1]  # label=1: 真实样本
    fake_uncertainties = uncertainties[labels == 0]  # label=0: 伪造样本
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 1. 直方图对比
    ax1.hist(real_uncertainties, bins=50, alpha=0.7, label='Real (label=1)', density=True, color='blue')
    ax1.hist(fake_uncertainties, bins=50, alpha=0.7, label='Fake (label=0)', density=True, color='red')
    ax1.axvline(0.01, color='green', linestyle='--', label='Threshold 0.01')
    ax1.set_xlabel('Uncertainty Score')
    ax1.set_ylabel('Density')
    ax1.set_title('Uncertainty Distribution Comparison\n(Real < 0.01, Fake > 0.01)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 箱线图对比
    data = [real_uncertainties, fake_uncertainties]
    labels_box = ['Real (label=1)', 'Fake (label=0)']
    box_plot = ax2.boxplot(data, labels=labels_box, patch_artist=True)
    
    # 设置颜色
    colors = ['lightblue', 'lightcoral']
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
    
    ax2.set_ylabel('Uncertainty Score')
    ax2.set_title('Uncertainty Distribution Box Plot')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"分布对比图已保存至: {save_path}")
    
    plt.show()
    
    return fig

def calculate_statistics(results):
    """
    计算两个类别的不确定性分数统计信息
    修正：根据您的观察调整标签解释
    """
    uncertainties = results['uncertainties']
    labels = results['labels']
    
    # 根据您的观察，label=1是真实样本，label=0是伪造样本
    real_uncertainties = uncertainties[labels == 1]  # label=1: 真实样本
    fake_uncertainties = uncertainties[labels == 0]  # label=0: 伪造样本
    
    # 计算统计信息
    real_stats = {
        'count': len(real_uncertainties),
        'mean': np.mean(real_uncertainties),
        'std': np.std(real_uncertainties),
        'min': np.min(real_uncertainties),
        'max': np.max(real_uncertainties),
        'median': np.median(real_uncertainties),
        'q25': np.percentile(real_uncertainties, 25),
        'q75': np.percentile(real_uncertainties, 75)
    }
    
    fake_stats = {
        'count': len(fake_uncertainties),
        'mean': np.mean(fake_uncertainties),
        'std': np.std(fake_uncertainties),
        'min': np.min(fake_uncertainties),
        'max': np.max(fake_uncertainties),
        'median': np.median(fake_uncertainties),
        'q25': np.percentile(fake_uncertainties, 25),
        'q75': np.percentile(fake_uncertainties, 75)
    }
    
    return {
        'real': real_stats,  # label=1
        'fake': fake_stats   # label=0
    }

def print_statistics_report(results, save_statistics_path='statistics_report.txt'):
    """
    打印详细的统计报告并保存到文件
    修正：更新标签解释
    """
    stats = calculate_statistics(results)
    
    report = f"""
WePe 检测器不确定性统计报告
==========================================
总样本数: {len(results['uncertainties'])}
真实样本数 (label=1): {stats['real']['count']}
伪造样本数 (label=0): {stats['fake']['count']}

真实样本不确定性统计 (label=1):
------------------------------------------
样本数量: {stats['real']['count']}
平均值: {stats['real']['mean']:.6f}
标准差: {stats['real']['std']:.6f}
最小值: {stats['real']['min']:.6f}
最大值: {stats['real']['max']:.6f}
中位数: {stats['real']['median']:.6f}
25%分位数: {stats['real']['q25']:.6f}
75%分位数: {stats['real']['q75']:.6f}

伪造样本不确定性统计 (label=0):
------------------------------------------
样本数量: {stats['fake']['count']}
平均值: {stats['fake']['mean']:.6f}
标准差: {stats['fake']['std']:.6f}
最小值: {stats['fake']['min']:.6f}
最大值: {stats['fake']['max']:.6f}
中位数: {stats['fake']['median']:.6f}
25%分位数: {stats['fake']['q25']:.6f}
75%分位数: {stats['fake']['q75']:.6f}

统计差异:
------------------------------------------
均值差异: {abs(stats['real']['mean'] - stats['fake']['mean']):.6f}
中位数差异: {abs(stats['real']['median'] - stats['fake']['median']):.6f}
重叠程度: {calculate_overlap(stats['real'], stats['fake']):.2f}%

注意:
------------------------------------------
根据您的观察，标签含义为:
- label=1: 真实样本 (real)
- label=0: 伪造样本 (fake)

真实样本不确定性大多在0.01以下，伪造样本不确定性大多在0.01以上
==========================================
"""
    # 打印到控制台
    print(report)
    
    # 保存到文件
    with open(save_statistics_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"统计报告已保存至: {save_statistics_path}")
    
    return stats

def calculate_overlap(real_stats, fake_stats):
    """
    计算两个分布的重叠程度
    """
    # 简单的重叠估计：基于均值±标准差的重叠区域
    real_range = (real_stats['mean'] - real_stats['std'], real_stats['mean'] + real_stats['std'])
    fake_range = (fake_stats['mean'] - fake_stats['std'], fake_stats['mean'] + fake_stats['std'])
    
    # 计算重叠区间
    overlap_start = max(real_range[0], fake_range[0])
    overlap_end = min(real_range[1], fake_range[1])
    
    if overlap_start > overlap_end:
        return 0
    
    # 计算重叠比例（简化估计）
    real_interval = real_range[1] - real_range[0]
    fake_interval = fake_range[1] - fake_range[0]
    overlap_length = overlap_end - overlap_start
    
    # 返回最大可能的重叠比例
    max_overlap = min(overlap_length / real_interval, overlap_length / fake_interval) * 100
    
    return max_overlap
    
def analyze_threshold_performance(results, candidate_thresholds=None):
    """
    分析多个候选阈值下的性能
    """
    if candidate_thresholds is None:
        # 基于您的数据分布设置候选阈值
        candidate_thresholds = [0.005, 0.008, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]
    
    uncertainties = results['uncertainties']
    labels = results['labels']
    
    print(f"\n=== 候选阈值性能分析 ===")
    best_f1 = 0
    best_threshold = 0
    
    for threshold in candidate_thresholds:
        predictions = (uncertainties > threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(labels, predictions)
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        eer = (fpr + fnr) / 2
        
        print(f"阈值 {threshold:.3f}: 准确率={accuracy:.4f}, F1={f1:.4f}, EER={eer:.4f}")
        print(f"          精确率={precision:.4f}, 召回率={recall:.4f}")
        print(f"          混淆矩阵: TP={tp}, FP={fp}, TN={tn}, FN={fn}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"\n最佳阈值: {best_threshold:.3f}, 最佳F1: {best_f1:.4f}")
    
    return best_threshold, best_f1