import torch
import torch.nn as nn
import torchaudio.transforms as T
import torchaudio.functional as F
import random
import math

class PseudoAnomalyAugmentor(nn.Module):
    """
    用于 Stage 2: 将真实语音转化为伪异常 (Pseudo-Anomaly)。
    目标：制造分布外的负样本 (Label = 0)。
    """
    def __init__(self, sample_rate=16000):
        super().__init__()
        self.sr = sample_rate
        
        # 1. 编解码/滤波 (Codec/Filter)
        # 模拟电话信道 (300-3400Hz)
        self.resample_8k = T.Resample(orig_freq=sample_rate, new_freq=8000)
        self.resample_back = T.Resample(orig_freq=8000, new_freq=sample_rate)
        # G.711 u-law 模拟 (降低位深和质量)
        self.mulaw_encode = T.MuLawEncoding(quantization_channels=128) # 甚至可以更低，比如 128
        self.mulaw_decode = T.MuLawDecoding(quantization_channels=128)

        # 2. 破坏性增强 (Destructive)
        # 极端变调 (Pitch Shift)
        self.pitch_shift_up = T.PitchShift(sample_rate, n_steps=6)  # 升6个半音
        self.pitch_shift_down = T.PitchShift(sample_rate, n_steps=-6) # 降6个半音

    def forward(self, waveform):
        """
        输入: (C, T) 或 (1, T)
        输出: (C, T) 且具有明显的非自然特征
        """
        # 随机选择一种破坏方式
        aug_type = random.choice(['noise', 'chopping', 'time_stretch', 'pitch_shift', 'codec_destroy'])
        
        # 确保维度
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        if aug_type == 'noise':
            return self._add_heavy_noise(waveform)
        elif aug_type == 'chopping':
            return self._chopping(waveform)
        elif aug_type == 'time_stretch':
            return self._time_stretch(waveform)
        elif aug_type == 'pitch_shift':
            return self._extreme_pitch(waveform)
        elif aug_type == 'codec_destroy':
            return self._codec_destroy(waveform)
        
        return waveform

    def _add_heavy_noise(self, wav):
        # 纯高斯噪声 或 模拟环境音
        noise_level = random.uniform(0.1, 0.5) # 很大的噪声
        noise = torch.randn_like(wav)
        return wav + noise_level * noise

    def _chopping(self, wav):
        # 极短片段随机拼接 (Temporal shuffling)
        # 将音频切成 N 块，然后打乱顺序拼回去
        B, T = wav.shape
        num_segments = random.randint(5, 20)
        segment_len = T // num_segments
        
        segments = list(torch.split(wav, segment_len, dim=-1))
        random.shuffle(segments)
        
        # 拼接
        new_wav = torch.cat(segments, dim=-1)
        # 修正长度 (因为 split 可能导致最后一块长度不同，这里做简单截断或padding)
        if new_wav.shape[-1] > T:
            new_wav = new_wav[..., :T]
        elif new_wav.shape[-1] < T:
            new_wav = F.pad(new_wav, (0, T - new_wav.shape[-1]))
            
        return new_wav

    def _time_stretch(self, wav):
        # 模拟变速 (不保调，类似于磁带快放/慢放，这是最不自然的)
        # 0.5x (慢) 到 2.0x (快)
        speed = random.choice([0.5, 0.6, 1.8, 2.0]) 
        # 使用 Resample 模拟变速效果：
        # 如果要把速度变快(2.0)，相当于把 sr*2 的采样率当做 sr 播放，采样点变少
        new_freq = int(self.sr * speed)
        resampler = T.Resample(orig_freq=self.sr, new_freq=new_freq)
        wav_stretched = resampler(wav)
        
        # 强制恢复原始长度 (Resample 会改变长度)
        target_len = wav.shape[-1]
        curr_len = wav_stretched.shape[-1]
        
        if curr_len > target_len:
            wav_stretched = wav_stretched[..., :target_len]
        else:
            wav_stretched = torch.nn.functional.pad(wav_stretched, (0, target_len - curr_len))
            
        return wav_stretched

    def _extreme_pitch(self, wav):
        # 极端变调但保速
        if random.random() < 0.5:
            return self.pitch_shift_up(wav)
        else:
            return self.pitch_shift_down(wav)

    def _codec_destroy(self, wav):
        # 强限带 + 低质量量化
        # 1. Bandpass filter via resampling
        wav = self.resample_8k(wav)
        # 2. Mu-law quantization (destroy detail)
        wav = self.mulaw_encode(wav)
        wav = self.mulaw_decode(wav)
        # 3. Back to 16k
        wav = self.resample_back(wav)
        
        # Fix length
        if wav.shape[-1] < 100: return torch.randn_like(wav) # 防止意外空音频
        return wav
    
class ASVspoofDomainAugmentation(torch.nn.Module):
    """
    模拟 ASVspoof LA 数据集的电话信道特征：
    1. 降采样到 8kHz (模拟电话带宽限制)
    2. G.711 编解码 (u-law 或 a-law)
    3. 升采样回 16kHz (为了适配 WavLM 输入)
    """
    def __init__(self, orig_freq=16000, target_freq=8000):
        super().__init__()
        self.resample_down = T.Resample(orig_freq=orig_freq, new_freq=target_freq)
        self.resample_up = T.Resample(orig_freq=target_freq, new_freq=orig_freq)
        
        # G.711 编解码模拟
        self.ulaw_encode = T.MuLawEncoding(quantization_channels=256)
        self.ulaw_decode = T.MuLawDecoding(quantization_channels=256)
        
        # 虽然 PyTorch 没有直接的 ALaw transform，但在模拟电话音质时，
        # Mu-law 已经能很好地破坏高频细节并引入量化噪声。
        
    def forward(self, waveform):
        # 输入 waveform shape: (C, T) 或 (T,)
        
        # 1. 模拟电话带宽：降采样到 8k
        # 这会切掉 4kHz 以上的所有高频信息，这是 ASVspoof 最显著的特征
        wav_low = self.resample_down(waveform)
        
        # 2. 模拟编解码器 (Codec) 噪声
        # 随机选择是否应用 G.711 u-law 压缩 (模拟电话传输的量化损失)
        if random.random() < 0.8: # 80% 的概率应用编解码噪声，20% 只做降采样
            wav_coded = self.ulaw_decode(self.ulaw_encode(wav_low))
        else:
            wav_coded = wav_low

        # 3. 恢复采样率到 16k (为了输入模型)
        wav_restored = self.resample_up(wav_coded)
        
        # 由于重采样可能会导致长度微小变化，做一下长度对齐
        # 这里只做简单的切片，后续 Dataset 会有更严格的 padding/cutting
        if wav_restored.shape[-1] > waveform.shape[-1]:
             wav_restored = wav_restored[..., :waveform.shape[-1]]
             
        return wav_restored
    
import torch
import torch.nn as nn
import torchaudio.transforms as T
import torch.nn.functional as F
import random
import numpy as np

class ASV2021_MLAAD_Augmentor(nn.Module):
    def __init__(self, sample_rate=16000):
        super().__init__()
        self.sr = sample_rate
        
        # --- 针对 ASV2021 LA/DF: 编解码模拟 ---
        self.mulaw_encode = T.MuLawEncoding(quantization_channels=128) # 模拟 LA 的 A-law/Mu-law
        self.mulaw_decode = T.MuLawDecoding(quantization_channels=128)
        
        # # --- 针对 ASV2021 PA: 频率响应与混响模拟 ---
        # # 简单的线性滤波器模拟麦克风频响差异
        # self.lowpass = T.Vad(sample_rate=sample_rate) # 借用VAD库或简单的Biquad
        
        # --- 针对 MLAAD: 神经声码器伪影模拟 ---
        # 模拟 MelGAN/HifiGAN 常见的高频金属音/毛刺
        # 我们用简单的加性高频噪声或共振峰破坏来模拟
        
    def forward(self, waveform):
        """
        输入: (B, C, T) 或 (C, T)
        输出: fake_wav, mask_wav (1=Modified, 0=Original)
        """
        if waveform.dim() == 2:
            waveform = waveform.unsqueeze(0)
        
        B, C, D = waveform.shape
        device = waveform.device
        
        fake_wav = waveform.clone()
        mask = torch.zeros((B, D), device=device) # 必须与 waveform 同样在 GPU

        # 对 Batch 中的每个样本独立选择增强策略
        for b in range(B):
            # 随机选择一种针对特定数据集的伪造手段
            aug_type = random.choice([
                'partial_codec',    # 针对 ASV21 LA/DF
                'partial_telephony',# 针对 ASV21 LA
                # 'partial_reverb',   # 针对 ASV21 PA
                'partial_vocoder',  # 针对 MLAAD
                'partial_silence'   # 通用：拼接/Masking
            ])
            
            # 随机确定局部区域 (Partial Segment)
            # 长度：总长的 10% 到 50%
            seg_len = int(D * random.uniform(0.6,0.8))
            start = random.randint(0, D - seg_len)
            end = start + seg_len
            
            # 提取该片段
            segment = fake_wav[b, :, start:end] # (C, L)
            
            # --- 执行增强 ---
            if aug_type == 'partial_codec':
                # 模拟低比特率量化 (ASV21 LA/DF)
                # 先压扁再还原，产生量化噪声
                aug_seg = self.mulaw_encode(segment)
                aug_seg = self.mulaw_decode(aug_seg)
                fake_wav[b, :, start:end] = aug_seg
                mask[b, start:end] = 1.0
                
            elif aug_type == 'partial_telephony':
                # 模拟电话带宽限制 (ASV21 LA)
                # 下采样到 8k 再上采样回 16k，丢失高频信息
                orig_len = segment.shape[-1]
                down = T.Resample(self.sr, 8000).to(device)(segment)
                up = T.Resample(8000, self.sr).to(device)(down)
                # 长度修正
                if up.shape[-1] != orig_len:
                    up = F.pad(up, (0, orig_len - up.shape[-1]))
                fake_wav[b, :, start:end] = up
                mask[b, start:end] = 1.0

            elif aug_type == 'partial_vocoder':
                # 模拟神经声码器伪影 (MLAAD)
                # 通常表现为高频的金属音或周期性伪影
                # 我们注入微弱的高频正弦波或加权噪声
                noise = torch.randn_like(segment) * 0.1
                # 制造周期性纹理
                t_vec = torch.arange(segment.shape[-1], device=device).float()
                metallic = torch.sin(2 * np.pi * t_vec / 10) * 0.02 # 高频纹理
                fake_wav[b, :, start:end] = segment + noise + metallic
                mask[b, start:end] = 1.0
                
            elif aug_type == 'partial_silence':
                # 模拟拼接检测中的 Zero-out (通用)
                fake_wav[b, :, start:end] = 0.0
                mask[b, start:end] = 1.0

        return fake_wav, mask.unsqueeze(1) # Mask: (B, 1, T)