import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader, WeightedRandomSampler
import numpy as np

from myutils.datasets.audio.LibriSpeech_ds import LibriSpeechAdapter
from myutils.datasets.audio._chime import CHiME5LocalDataset
from myutils.datasets.audio._voxceleb import VoxParquetBytesDataset
from myutils.datasets.audio.common_voice import Common_voice_dataset

class DomainIndexedDataset(Dataset):
    """
    包装器：将原始 Dataset 的字符串 domain_label 转换为整数 ID (0, 1, 2...)
    同时作为一个中间层，确保 audio 没有任何梯度残留。
    """
    def __init__(self, dataset, domain_id: int):
        self.dataset = dataset
        self.domain_id = domain_id

    def __len__(self):
        return len(self.dataset)

    def __getattr__(self, name):
        # 允许直接访问被包装 Dataset 的属性 (如 self.dataset.stage)
        return getattr(self.dataset, name)

    def __setattr__(self, name, value):
        # 允许设置被包装 Dataset 的属性
        if name in ['dataset', 'domain_id']:
            object.__setattr__(self, name, value)
        else:
            setattr(self.dataset, name, value)

    def __getitem__(self, idx):
        batch = self.dataset[idx]
        
        # 1. 强制覆盖 domain_label 为整数 (LongTensor)
        # Domain Discriminator 的 CrossEntropy 需要 Long 类型
        batch['domain_label'] = torch.tensor(self.domain_id, dtype=torch.long)
        
        # 2. 双重保险：确保 audio detached
        if batch['audio'].requires_grad:
            batch['audio'] = batch['audio'].detach()
            
        return batch
    
class MultiSourceFusionFactory:
    def __init__(self, args, stage,asv_aug = False, pseudo_aug = False):
        """
        args: 包含各数据集路径的配置对象
        """
        # 定义域 ID 映射
        self.domain_map = {
            'librispeech': 0,
            'common_voice': 1,
            'voxceleb': 2,
            'chime': 3
        }
        
        print(">>> Initializing Multi-Source Datasets...")
        
        # 1. 实例化各个数据集 (默认 Stage 1)
        self.ds_libri = LibriSpeechAdapter(
            asv_aug=asv_aug, pseudo_aug=pseudo_aug, stage=stage
        )
        
        self.ds_cv = Common_voice_dataset(
            processed_root_path=args.cv_path, # e.g. "/home/zyz/data/..."
            aug=False, asv_aug=asv_aug, pseudo_aug=pseudo_aug, stage=stage, lang_codes='all'
        )
        
        self.ds_vox = VoxParquetBytesDataset(
            pseudo_aug=pseudo_aug, stage=stage
        )
        
        self.ds_chime = CHiME5LocalDataset(
            root=args.chime_path, # e.g. "/path/to/chime5"
            pseudo_aug=pseudo_aug, stage=stage
        )

        # 2. 使用 Wrapper 赋予它们整数 ID
        self.wrapped_datasets = [
            DomainIndexedDataset(self.ds_libri, self.domain_map['librispeech']),
            DomainIndexedDataset(self.ds_cv,    self.domain_map['common_voice']),
            DomainIndexedDataset(self.ds_vox,   self.domain_map['voxceleb']),
            DomainIndexedDataset(self.ds_chime, self.domain_map['chime'])
        ]
        
        # 3. 物理融合
        self.concat_dataset = ConcatDataset(self.wrapped_datasets)
        
        # 4. 计算不平衡采样权重 (关键步骤！)
        self.sampler = self._make_balanced_sampler()
        
    def _make_balanced_sampler(self):
        """
        计算 WeightedRandomSampler，使得每个 Batch 中各域的数据量大致相等。
        如果不做这一步，Libri/CV 会淹没 CHiME。
        """
        print("Calculating sampler weights for balanced domain training...")
        dataset_counts = [len(ds) for ds in self.wrapped_datasets]
        total_count = sum(dataset_counts)
        num_domains = len(dataset_counts)
        
        # 每个域的权重 = 总数 / (域数量 * 该域样本数)
        # 使得 该域权重 * 该域样本数 = 总数 / 域数量 (即每个域的总“影响力”相等)
        class_weights = [total_count / (num_domains * c) if c > 0 else 0 for c in dataset_counts]
        
        # 为每一个样本分配权重
        sample_weights = []
        for i, count in enumerate(dataset_counts):
            # 将对应域的权重重复 count 次
            sample_weights.extend([class_weights[i]] * count)
            
        return WeightedRandomSampler(
            weights=torch.as_tensor(sample_weights, dtype=torch.double),
            num_samples=total_count,
            replacement=True
        )

    def set_stage(self, stage):
        """
        统一切换所有子数据集的 Stage (Stage 1 <-> Stage 2)
        """
        print(f"Switching all datasets to {stage}...")
        # 因为 DomainIndexedDataset 设置了 __setattr__ 代理，
        # 直接设置 wrapped_ds.stage 就会修改到底层 dataset
        for ds in self.wrapped_datasets:
            ds.stage = stage

    def get_dataloader(self, batch_size=32, num_workers=8):
        return DataLoader(
            self.concat_dataset,
            batch_size=batch_size,
            sampler=self.sampler, # <--- 必须使用 Sampler
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True
        )