from pathlib import Path
import random

import torch
import torchaudio

from myutils.datasets.audio.Augment import PseudoAnomalyAugmentor


class CHiME5LocalDataset:
    """
    从本地加载已经下载并解压的 CHiME5 eval 集合
    """

    def __init__(self, root, pseudo_aug = False, stage='stage1',sample_rate=16000):
        """
        root: chime5_eval/ 解压后的目录
        """
        self.root = Path(root)
        assert self.root.exists(), f"路径不存在: {root}"
        self.stage = stage
        self.target_length = 48000
        self.sample_rate = sample_rate
        self.pseudo_aug = pseudo_aug
        self.pseudo_augmenter = PseudoAnomalyAugmentor(sample_rate=16000)
        # CHiME5 所有 wav 都在 wav/ 子目录下
        wav_dir = self.root / "audio/eval"
        assert wav_dir.exists(), f"未找到 audio 目录: {wav_dir}"

        # 递归搜索所有 wav 文件
        self.wav_files = list(wav_dir.rglob("*.wav"))

        print(f"[CHiME5] 共加载 {len(self.wav_files)} 个音频文件")

    def __len__(self):
        return len(self.wav_files)

    def __getitem__(self, idx):
        wav_path = self.wav_files[idx]
        waveform, sr = torchaudio.load(str(wav_path))
        if waveform.shape[0] > 1:
            waveform = waveform[0:1, :] # 变成 [1, T]
        # 自动重采样为 16k
        label = 1.0
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        waveform = cut_length(waveform=waveform,target_length=self.target_length)
        if self.pseudo_aug:
            if random.random() < 0.5:
                waveform = self.pseudo_augmenter(waveform)
                label = 0.0
        return {
            "audio": waveform.squeeze(0).detach(),                 # Tensor [1, T]
            "domain_label": "chime",
            "label": label,
        }

def cut_length(waveform,target_length):
    current_len = waveform.shape[1]
    if current_len > target_length:
        waveform = waveform[:, :target_length]
    elif current_len < target_length:
        padding = target_length - current_len
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    return waveform
# if __name__ == "__main__":
#     root = "/home/zyz/data/work2/Datasets/chime_10mb"   # 你解压的位置
#     ds = CHiME5LocalDataset(root=root,pseudo_aug=True, stage="stage2")

#     print("\n示例：")
#     sample = ds[0]
#     print(sample["domain_label"], sample["audio"].shape, sample["label"])