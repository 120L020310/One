import argparse
import os
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F

from Oneclass_XLSR_lit import ALDA_OneClass_Lit
from config.config import get_cfg_defaults
from data.make_dataset import make_data
from utils_work2 import compute_metrics_improved, find_optimal_threshold_by_eer, plot_distribution_comparison, plot_results

class MCDropoutDetector:
    def __init__(self, model, device='cuda', num_passes=8, dropout_rate=None):
        """
        MC Dropout 检测器
        :param model: 训练好的模型
        :param num_passes: 前向传播的次数 (N)
        :param dropout_rate: (可选) 如果模型原本dropout太低，可以强制修改 dropout rate
        """
        self.model = model
        self.device = device
        self.num_passes = num_passes
        self.model.to(device)
        
        # 关键步骤：设置模型状态
        # 1. 全局设为 eval (锁定 BatchNorm 的均值和方差，这点非常重要！)
        self.model.eval()
        
        # 2. 局部开启 Dropout (只让 Dropout 层处于 train 模式)
        self._enable_dropout_only(self.model, dropout_rate)
        
        print(f"✅ MC Dropout 检测器就绪: Forward Pass = {self.num_passes} 次")

    def _enable_dropout_only(self, model, force_rate=None):
        """
        递归地查找所有 Dropout 层并激活它们，同时保持 BN 层冻结
        """
        count = 0
        for m in model.modules():
            if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
                m.train() # 激活 Dropout
                if force_rate is not None:
                    m.p = force_rate # 强制修改概率
                count += 1
        print(f"   已激活 {count} 个 Dropout 层 (Rate: {force_rate if force_rate else 'Original'})")

    def evaluate_on_dataset(self, dataloader, method="variance", thresholds=None):
        """
        在数据集上评估
        """
        print("开始评估 MC Dropout 性能...")
        
        all_uncertainties = []
        all_labels = []
        
        # 不需要 torch.no_grad()，因为我们需要 Dropout 随机性
        # 但我们不需要计算梯度，所以还是要加 no_grad 来省显存
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(dataloader)):
                # 数据解包
                if isinstance(batch, dict):
                    audio_data = batch['audio']
                    labels = batch['label']
                elif isinstance(batch, list):
                    audio_data = torch.stack([item['audio'] for item in batch])
                    labels = torch.tensor([item['label'] for item in batch])
                
                audio = audio_data.to(self.device)
                
                # --- MC Dropout 核心计算逻辑 ---
                batch_uncertainties = self._compute_batch_uncertainty(audio)
                
                all_uncertainties.extend(batch_uncertainties.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                if batch_idx == 10: # 调试用
                     break
        
        all_uncertainties = np.array(all_uncertainties)
        all_labels = np.array(all_labels)
        
        print(f"处理完成! 样本数: {len(all_uncertainties)}")
        print(f"不确定性均值: {all_uncertainties.mean():.4f}")
        
        # 复用你之前的指标计算函数
        results = compute_metrics_improved(all_uncertainties, all_labels, thresholds)
        results['uncertainties'] = all_uncertainties
        results['labels'] = all_labels
        
        return results

    def _compute_batch_uncertainty(self, inputs):
        """
        对一个 Batch 进行 N 次前向传播并计算方差
        """
        features_list = []
        
        # 1. 进行 N 次前向传播
        for _ in range(self.num_passes):
            # 注意：这里假设你的模型直接返回特征 tensor
            # 如果模型返回的是 dict (如 {'final_feat': ...})，请自行调整取值
            outputs = self.model(inputs) 
            
            # 兼容性处理：如果输出是字典或元组，取特征向量
            if isinstance(outputs, dict):
                features = outputs.get('final_feat', list(outputs.values())[0])
            elif isinstance(outputs, (tuple, list)):
                features = outputs[0]
            else:
                features = outputs
                
            features_list.append(features.mean(1)) # Shape: (B, D)
            
        # 2. 堆叠: (N_passes, B, D)
        stacked_features = torch.stack(features_list)
        
        # 3. 计算不确定性 (Uncertainty)
        # 假设 Real 样本非常稳定，方差小；Fake 样本方差大
        
        # 方法 A: 欧氏距离方差 (Variance of features)
        # 对 N 次结果求方差，然后在特征维度求均值 -> (B,)
        # variance = torch.var(stacked_features, dim=0).mean(dim=1)
        # v_min = variance.min()
        # v_max = variance.max()
        # norm_v = (variance - v_min) / (v_max - v_min + 1e-9)
        
        # 方法 B (可选): 余弦相似度
        # 计算每次 output 与 均值 center 的余弦距离
        center = torch.mean(stacked_features, dim=0) # (B, D)
        center_norm = F.normalize(center, p=2, dim=1)
        sims = []
        for i in range(self.num_passes):
            feat_norm = F.normalize(stacked_features[i], p=2, dim=1)
            
            # 3. 使用 matmul 计算逐行余弦相似度
            # 变形: [64, 1, 1024] @ [64, 1024, 1] -> [64, 1, 1]
            # 利用 matmul 的广播机制进行 Batch Dot Product
            sim = torch.matmul(feat_norm.unsqueeze(1), center_norm.unsqueeze(2)).squeeze()
            sims.append(sim)
        variance = 1.0 - torch.stack(sims).mean(dim=0)

        return variance

    def save_detection_results(self, results, save_path):
        # 复用你之前的保存逻辑
        df = pd.DataFrame({
            'uncertainty': results['uncertainties'],
            'label': results['labels'],
            'label_description': ['real' if label == 1 else 'fake' for label in results['labels']]
        })
        df.to_csv(save_path, index=False)
        return df
    
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

    sd = torch.load("/home/zyz/data/test_controlled_ex/1217_ALDA_oneclass/MultiView/ASV2021_LA/version_0/checkpoints/best-epoch=6-val-auc=0.9630.ckpt", map_location="cpu")["state_dict"]
    model.load_state_dict(sd)
    model = model.model.to(device=device)
    ds, dl = make_data(cfg.DATASET, args=args)
    ds_name = args.cfg.split("/")[1]
    # --- 修改开始: 使用 MC Dropout Detector ---
    print("🚀 正在初始化 MC Dropout 检测器...")
    detector = MCDropoutDetector(
        model, 
        device=device,
        num_passes=8,  # 建议设为 10 到 50 之间
        dropout_rate=0.2 # 如果你的模型本身 dropout 很小，可以尝试强制设为 0.1 或 0.2
    )
    
    # 定义输出路径
    output_dir = f"/home/zyz/work2_dir/1217_new_method1/mcdropout_{ds_name}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 评估循环
    for i in range(len(dl.test)):
        print(f"\n正在评估测试集: {i}")
        
        # 调用评估 (不再需要 method 参数，因为逻辑内置了)
        results = detector.evaluate_on_dataset(dl.test[i])
        
        # ... (后续的指标计算、绘图逻辑完全不用动，直接复用) ...
        
        # 1. 基于EER寻找最优阈值
        eer_threshold, eer, eer_accuracy, eer_f1 = find_optimal_threshold_by_eer(results)
        
        print(f"EER: {eer:.4f} @ Threshold: {eer_threshold:.6f}")
        
        # 保存结果
        detector.save_detection_results(results, f'{output_dir}/results_set_{i}.csv')
        
        # 绘图等 (复用你现有的函数)
        plot_distribution_comparison(results, f'{output_dir}/dist_set_{i}.png')
        plot_results(results, save_path=f'{output_dir}/roc_set_{i}.png')

    print("✅ MC Dropout 评估完成!")