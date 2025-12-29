import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

class RobustDataAugmentor(nn.Module):
    def __init__(self, p=0.8):
        super().__init__()
        self.p = p # 触发概率
        # 这里的 mask_param 需要根据你的特征维度调整，如果是原始音频，模拟的是Time Mask
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=15)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=35)

    def forward(self, audio):
        """
        Input: audio [B, T] or [B, 1, T]
        """
        if not self.training:
            # 测试时为了保持确定性，通常不随基变，但在你的Anomaly Detection逻辑里，
            # 测试时也需要这个增强来产生 feat_noisy 来和 feat_clean 对比！
            # 所以这里不做 if self.training 的拦截，或者由外部控制
            pass

        # 确保输入是 [B, 1, T] 用于插值
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
        
        B, C, T = audio.shape
        device = audio.device
        
        # --- 策略1: 模拟编解码/传输损失 (Resampling) ---
        # 这是一个非常强的Deepfake对抗手段。
        # 许多Vocoder产生的伪影在高频，降采样再升采样会抹除这些伪影。
        # Real样本对降采样鲁棒，Fake样本对降采样敏感。
        
        # 随机降采样因子 (0.5 ~ 0.9) -> 模拟 8k-14k Hz 采样率
        scale = 0.5 + 0.4 * torch.rand(B, 1, 1, device=device)
        
        # 由于每个样本scale不同，不能直接batch处理，为了效率我们统一采样一个scale
        # 或者为了更强的随机性，对整个batch用同一个scale (效率高)
        random_scale = 0.5 + 0.4 * torch.rand(1).item()
        
        # Downsample
        audio_low = F.interpolate(audio, scale_factor=random_scale, mode='linear', align_corners=False)
        # Upsample back to original length
        audio_noisy = F.interpolate(audio_low, size=T, mode='linear', align_corners=False)

        # --- 策略2: 加上微弱的加性噪声 (Additive Noise) ---
        # 增加一点底噪，防止模型过拟合于静音段
        noise = torch.randn_like(audio_noisy) * 0.01
        audio_noisy = audio_noisy + noise
        
        # --- 策略3: 随机幅度缩放 (Volume Perturbation) ---
        # 改变音量，Real样本应该不论音量大小特征都一致
        gain = 0.8 + 0.4 * torch.rand(B, 1, 1, device=device)
        audio_noisy = audio_noisy * gain

        return audio_noisy.squeeze(1)
    
class SafeRawAugmentor(nn.Module):
    def __init__(self, noise_intensity=0.05, mask_ratio=0.1):
        """
        参数:
        noise_intensity: 噪声强度 (0.01 ~ 0.1 比较合适)
        mask_ratio: 遮挡比例 (例如 0.1 表示遮挡 10% 的长度)
        """
        super().__init__()
        self.noise_intensity = noise_intensity
        self.mask_ratio = mask_ratio

    def forward(self, audio):
        """
        Input: audio [B, 48000]
        Output: audio_aug [B, 48000]
        """
        B, L = audio.shape
        device = audio.device
        
        # 复制一份，不修改原tensor
        audio_aug = audio.clone()

        # ---------------------------------------------------
        # 1. 随机幅度缩放 (Amplitude Scaling) - 模拟录音增益差异
        # ---------------------------------------------------
        # Real样本对音量变化应该是完全鲁棒的
        # 范围 [0.5, 1.5]
        scale = 0.5 + torch.rand(B, 1, device=device)
        audio_aug = audio_aug * scale

        # ---------------------------------------------------
        # 2. 加性高斯白噪声 (Additive White Gaussian Noise)
        # ---------------------------------------------------
        # 这是一个非常通用的物理增强，不涉及任何编码算法
        # 计算每个样本的能量，根据信噪比加噪
        # 为了高效，直接加绝对噪声，或者基于std加
        noise = torch.randn_like(audio_aug)
        
        # 动态计算每个样本的强度，保证噪声既有干扰性又不至于完全淹没信号
        # 这里使用信号的 RMS * intensity
        rms = torch.sqrt(torch.mean(audio_aug**2, dim=1, keepdim=True))
        # 加上微小的 epsilon 防止静音段报错
        noise_level = rms * self.noise_intensity
        
        audio_aug = audio_aug + noise * noise_level

        # ---------------------------------------------------
        # 3. 时域遮挡 (Time Domain Masking / Cutout)
        # ---------------------------------------------------
        # 强制模型利用上下文信息，而不是依赖某一段特定的伪影
        # ASVspoof 很多攻击是基于帧的，遮挡可以破坏这种连续性
        
        mask_len = int(L * self.mask_ratio)
        
        # 为每个样本生成一个随机起始点
        # start_index: [B]
        start_indices = torch.randint(0, L - mask_len, (B,), device=device)
        
        # 向量化遮挡：生成掩码矩阵
        # 这里用一个小技巧避免循环：创建一个 [B, L] 的 mask
        # 但考虑到显存，我们用循环给每个sample置零也很快，因为Batch不大
        # 为了追求极致的Torch写法：
        
        # 生成一个 [1, L] 的索引序列
        indices = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        # 生成 [B, 1] 的起始和结束
        starts = start_indices.unsqueeze(1)
        ends = starts + mask_len
        
        # 创建 mask: 在范围内为 0，其余为 1
        mask = ~((indices >= starts) & (indices < ends))
        
        # 应用遮挡
        audio_aug = audio_aug * mask.float()

        return audio_aug
    
import torch
import torch.nn as nn

class TransformerHardAugmentor(nn.Module):
    def __init__(self, noise_intensity=0.01, mask_ratio=0.15, span_len=3200):
        """
        参数:
        noise_intensity: 噪声强度 (0.01 左右)
        mask_ratio: 总遮挡比例
        span_len: 单个遮挡块的长度 (重要！)。
                  48000Hz下，3200点约为66ms。
                  Wav2Vec2的感受野很大，建议设大一点，比如 4000-8000 (100-200ms)
        """
        super().__init__()
        self.noise_intensity = noise_intensity
        self.mask_ratio = mask_ratio
        # 强制转换为int
        self.span_len = int(span_len) 

    def forward(self, audio):
        """
        Input: audio [B, 48000]
        """
        B, L = audio.shape
        device = audio.device
        audio_aug = audio.clone()

        # ---------------------------------------------------
        # 1. 非线性削波 (Clipping / Saturation) - 替代 Scaling
        # ---------------------------------------------------
        # 模拟麦克风录音过载。这改变了波形的分布形态（峰值变平），
        # LayerNorm 无法还原这种非线性变化，因此能产生有效的特征差异。
        # 随机选择截断阈值，例如 [0.7, 0.95]
        clip_threshold = 0.7 + 0.25 * torch.rand(B, 1, device=device)
        audio_aug = torch.clamp(audio_aug, min=-clip_threshold, max=clip_threshold)

        # ---------------------------------------------------
        # 2. Span Masking (大块遮挡) - 替代随机Mask
        # ---------------------------------------------------
        # 计算需要遮挡的总长度
        total_mask_len = int(L * self.mask_ratio)
        # 计算需要几个块
        num_spans = max(1, total_mask_len // self.span_len)

        # 生成掩码
        mask = torch.ones_like(audio_aug)
        
        for i in range(B):
            for _ in range(num_spans):
                # 随机选起点
                start = torch.randint(0, L - self.span_len, (1,)).item()
                end = start + self.span_len
                # 挖空
                mask[i, start:end] = 0.0
        
        audio_aug = audio_aug * mask

        # ---------------------------------------------------
        # 3. 加性噪声 (保持不变，作为基础扰动)
        # ---------------------------------------------------
        noise = torch.randn_like(audio_aug)
        rms = torch.sqrt(torch.mean(audio_aug**2, dim=1, keepdim=True))
        audio_aug = audio_aug + noise * rms * self.noise_intensity

        return audio_aug
    
import torch
import torch.nn as nn

class DigitalArtifactAugmentor(nn.Module):
    def __init__(self, codec_intensity=0.5):
        super().__init__()
        self.codec_intensity = codec_intensity

    def forward(self, audio):
        """
        模拟数字音频的非线性失真（量化噪声、μ-law压缩伪影）
        Input: [B, L]
        """
        B, L = audio.shape
        device = audio.device
        audio_aug = audio.clone()

        # ------------------------------------------------
        # 1. 模拟低比特率量化 (Bit-depth Reduction)
        # ------------------------------------------------
        # 很多 Deepfake 音频生成时精度不足。
        # 模拟将 16-bit 音频降低到 4-bit ~ 8-bit
        # 这种非线性阶梯状失真对 Real 破坏很大，但 Fake 可能本来就有。
        
        bits = 4 + 4 * torch.rand(B, 1, device=device) # 随机 4-8 bit
        scale = 2 ** bits
        
        # 量化再反量化
        audio_aug = torch.round(audio_aug * scale) / scale

        # ------------------------------------------------
        # 2. μ-law 压扩 (Mu-law Companding)
        # ------------------------------------------------
        # WaveNet 等早期声码器常用的处理，会带来特征性的量化底噪
        if torch.rand(1).item() > 0.5:
            mu = 255.0
            # Encode
            audio_sign = torch.sign(audio_aug)
            audio_abs = torch.abs(audio_aug)
            audio_aug = audio_sign * (torch.log1p(mu * audio_abs) / torch.log1p(torch.tensor(mu)))
            # Decode (带损失)
            audio_aug = torch.sign(audio_aug) * ((1 + mu)**torch.abs(audio_aug) - 1) / mu

        return audio_aug