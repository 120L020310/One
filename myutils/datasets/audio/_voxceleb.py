import glob
import io
import random
import pandas as pd
import torch
from torch.utils.data import Dataset
import torchaudio
from datasets import load_dataset
from myutils.datasets.audio.Augment import PseudoAnomalyAugmentor
from myutils.datasets.audio._chime import cut_length

class VoxParquetBytesDataset(Dataset):
    def __init__(self, pseudo_aug=False, stage='stage1', transform=None):
        self.stage = stage
        self.target_length = 48000
        # 获取所有 parquet 文件路径
        parquet_files = glob.glob("/home/zyz/data/work2/Datasets/voxceleb/*.parquet")
        
        # 【修改点】使用 load_dataset 进行内存映射加载，不占 RAM
        self.ds = load_dataset("parquet", data_files=parquet_files, split="train")
        
        self.pseudo_aug = pseudo_aug
        self.pseudo_augmenter = PseudoAnomalyAugmentor(sample_rate=16000)
        self.transform = transform
        

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        audio_entry = row["audio"]
        # 取出 bytes
        waveform = torch.from_numpy(audio_entry["array"]).float()
        sr = audio_entry["sampling_rate"]
        
        # datasets 解码出来的通常是 (T,)，我们需要转为 (1, T)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
            waveform = waveform[0:1, :] # 变成 [1, T]        
        label=1.0
        # 可选 transform
        if self.transform:
            waveform = self.transform(waveform)
        waveform = cut_length(waveform=waveform,target_length=self.target_length)
        if self.pseudo_aug:
            if random.random() < 0.5:
                waveform = self.pseudo_augmenter(waveform)
                label = 0.0
        return {
            "audio": waveform.squeeze(0).detach(),               # (channels, time)
            "label": label,
            "domain_label": "voxceleb"                        # 可选：保留 id
        }


# import glob
# from torch.utils.data import DataLoader

# dataset = VoxParquetBytesDataset()

# loader = DataLoader(dataset, batch_size=4, shuffle=True)

# batch = next(iter(loader))
# print(batch["audio"].shape)  # (B, C, T)
# print(batch["label"])
# print(batch["id"])
