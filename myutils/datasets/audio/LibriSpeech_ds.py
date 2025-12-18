import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import random
import os
import pickle
from tqdm import tqdm
from torch.utils.data import Dataset
import pickle

from myutils.datasets.audio.Augment import PseudoAnomalyAugmentor
from myutils.datasets.audio.common_voice import ASVspoofDomainAugmentation # <-- 导入 pickle 库
# ==========================================
# 1. 定义增强模块
# ==========================================
class WaveformAugmentations(nn.Module):
    """
    内存级波形增强模块：
    包含：变速(拉伸)、模拟压缩/低码率、带宽限制(重采样)、加噪声。
    """
    def __init__(self, sample_rate=16000, augment_prob=0.5):
        super().__init__()
        self.sample_rate = sample_rate
        self.prob = augment_prob

        # 1. 模拟低带宽/重采样 (Downsample -> Upsample)
        # 模拟电话(8k) 或 广播(16k->12k)
        self.downsample_factors = [8000, 12000] 

        # 2. 变速/拉伸范围
        self.speed_range = (0.9, 1.1)

    def add_noise(self, waveform, snr_db_min=10, snr_db_max=30):
        """加入高斯白噪声"""
        if random.random() > self.prob:
            return waveform
            
        # 计算信号能量
        signal_power = waveform.pow(2).mean()
        if signal_power == 0:
            return waveform
            
        # 随机选取 SNR
        snr_db = random.uniform(snr_db_min, snr_db_max)
        snr = 10 ** (snr_db / 10)
        
        # 计算噪声功率和缩放比例
        noise_power = signal_power / snr
        noise_scale = torch.sqrt(noise_power)
        
        noise = torch.randn_like(waveform) * noise_scale
        return waveform + noise

    def simulate_compression(self, waveform):
        """
        模拟压缩/容器变换/低比特率
        方法：使用 Mu-law 编码再解码，或者降低位深 (Quantization)
        """
        if random.random() > self.prob:
            return waveform

        # 方式 A: Mu-law 编码 (模拟 G.711 电话压缩效果，会有明显的量化噪声)
        # quantization_channels 通常为 256
        encoded = torchaudio.functional.mu_law_encoding(waveform, quantization_channels=256)
        decoded = torchaudio.functional.mu_law_decoding(encoded, quantization_channels=256)
        return decoded

    def resample_quality_drop(self, waveform):
        """
        模拟低采样率录音设备 (降采样再升采样)
        """
        if random.random() > self.prob:
            return waveform
            
        target_sr = random.choice(self.downsample_factors)
        
        # 降采样
        resampler_down = T.Resample(self.sample_rate, target_sr, dtype=waveform.dtype)
        # 升采样回原频率 (为了保持 tensor 大小一致性)
        resampler_up = T.Resample(target_sr, self.sample_rate, dtype=waveform.dtype)
        
        return resampler_up(resampler_down(waveform))

    def time_stretch(self, waveform):
        """
        变速 (Speed Perturbation)。这会改变波形的长度 T。
        """
        if random.random() > self.prob:
            return waveform
            
        speed = random.uniform(self.speed_range[0], self.speed_range[1])
        
        # 纯 PyTorch 方法模拟变速：
        # 使用 Resample 模拟变速。
        # 如果 speed > 1 (变快)，则新的波形点数会减少。
        # 如果 speed < 1 (变慢)，则新的波形点数会增加。
        
        # 目标采样率保持 self.sample_rate，但我们假装原始采样率是 self.sample_rate * speed
        # 这就是 torchaudio Speed Perturbation 的常见实现方式
        
        # 为了兼容性，使用 torchaudio.transforms.Resample
        resampler = T.Resample(
            orig_freq=int(self.sample_rate / speed), # 假装原始频率
            new_freq=self.sample_rate, 
            dtype=waveform.dtype
        )
        
        # Resample 期望 [C, T] 或 [T]，我们假设输入是 [T]
        if waveform.ndim == 1:
            return resampler(waveform.unsqueeze(0)).squeeze(0)
        else:
            return resampler(waveform)
# --- 3. LibriSpeech 数据集处理 (带缓存优化版) ---
class LibriSpeechAdapter(Dataset):
    """
    封装 LibriSpeechForMemoryBank，使其输出格式匹配 (张量, 标签)。
    """
    def __init__(self, asv_aug = False, pseudo_aug = False, stage='stage1'):
        self.original_dataset = LibriSpeechForMemoryBank(
            augment=False,
            data_root="/home/zyz/data/work2/Datasets/LibriSpeech", 
            subset="train-clean-360",
            segment_length_sec=3 # 4秒片段
        )
        self.stage = stage
        self.asv_aug = asv_aug
        self.pseudo_aug = pseudo_aug
        self.pseudo_augmenter = PseudoAnomalyAugmentor(sample_rate=16000)
        self.domain_augmenter = ASVspoofDomainAugmentation(orig_freq=16000)
    def __len__(self):
        return len(self.original_dataset)
    def __getitem__(self, idx):
        # 假设原数据集只返回一个张量 (即原始波形张量)
        batch={}
        input_tensor = self.original_dataset[idx]
        label = 1.0
        # Stage 2 负样本生成
        if self.pseudo_aug:
            if random.random() < 0.5:
                input_tensor = self.pseudo_augmenter(input_tensor)
                label = 0.0
        
        # 如果是 Real，可以选择性加域增强
        if label == 1.0 and self.asv_aug:
             if random.random() < 0.5:
                 input_tensor = self.domain_augmenter(input_tensor)
        batch["audio"]=input_tensor.squeeze(0) if input_tensor.dim()==2 else input_tensor
        batch["domain_label"]="librispeech"
        batch["label"]=label
        # 返回 (输入张量, 标签)
        return batch
class LibriSpeechSpeakerID(Dataset):
    """
    用于说话人识别的 LibriSpeech 数据集。
    - 增加缓存机制，跳过重复的样本解析和列表填充步骤。
    """
    def __init__(self, data_root, subset="train-clean-360", sample_rate=16000, n_mels=80, segment_length_sec=3):
        super().__init__()
        
        self.data_root = data_root
        self.subset = subset
        self.sample_rate = sample_rate
        self.segment_length_samples = sample_rate * segment_length_sec
        self.librispeech_raw = torchaudio.datasets.LIBRISPEECH(root=data_root, url=subset, download=False)
        # 定义缓存文件路径
        cache_filename = f"{subset}_samples_cache_sr{sample_rate}_len{segment_length_sec}s.pkl"
        self.cache_path = os.path.join(data_root, cache_filename)
        
        # 初始化转换器 (不管是否使用缓存，转换器都需要)
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate, n_fft=1024, win_length=400, hop_length=160, n_mels=n_mels
        )
        self.amplitude_to_db = T.AmplitudeToDB()

        self.speaker_to_int = {}
        self.samples = []

        # --- 缓存检查和加载 ---
        if os.path.exists(self.cache_path):
            print(f"检测到缓存文件: {self.cache_path}，正在加载...")
            try:
                with open(self.cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                    self.samples = cache_data['samples']
                    self.speaker_to_int = cache_data['speaker_to_int']
                    print(f"缓存加载成功。总样本数: {len(self.samples)}")
            except Exception as e:
                print(f"加载缓存失败 ({e})，将重新解析数据集。")
                self._parse_and_cache_dataset()
        else:
            self._parse_and_cache_dataset()


    def _parse_and_cache_dataset(self):
        """
        执行耗时的 LibriSpeech 数据集解析、说话人映射创建和样本列表填充。
        这一次，我们缓存文件路径，而不是波形 Tensor。
        """
        print("未找到缓存，开始解析 LibriSpeech 数据集...")
        
        all_speaker_ids = set()
        temp_samples_metadata = [] # 临时存储 (file_path, speaker_id, sr)
        
        print(f"正在遍历 {len(self.librispeech_raw)} 个样本...")
        
        for i in tqdm(range(len(self.librispeech_raw)), desc=f"解析 {self.subset} 样本"):
            # LIBRISPEECH 默认返回 (waveform, sample_rate, transcript, speaker_id, chapter_id, utterance_id)
            # 为了获取路径，我们必须依赖 LIBRISPEECH 内部机制，它通常与 FLAC 文件名相关
            
            # --- 重要的修改点：获取路径和元数据 ---
            # 直接调用 self.librispeech_raw[i] 仍然会加载波形，造成 OOM。
            # 我们需要获取路径。torchaudio 的 LIBRISPEECH 类的 `_load_audio` 方法依赖于
            # `self._walker` 或 `self.data` 属性（取决于版本）。
            
            try:                
                # 依赖于文件结构的手动遍历：
                item = self.librispeech_raw._walker[i] # 假设 walker 包含文件路径元组
                # item 结构通常是 (speaker_id, chapter_id, utterance_id, fileid_flac, fileid_txt)
                speaker_id = item.split("-")[0]
                chapter_id = item.split("-")[1]
                utterance_id = item.split("-")[2]
                
                relative_path = os.path.join(str(speaker_id), str(chapter_id), f"{speaker_id}-{chapter_id}-{utterance_id}.flac")
                full_path = os.path.join(self.data_root, "LibriSpeech", self.subset, relative_path)
                
                # 假设采样率是固定的 16000
                sr = self.sample_rate
                
                all_speaker_ids.add(speaker_id)
                temp_samples_metadata.append((full_path, speaker_id, sr))
                
            except Exception as e:
                # 某些样本可能损坏或路径构造失败
                continue

        sorted_speaker_ids = sorted(list(all_speaker_ids))
        self.speaker_to_int = {speaker_id: i for i, speaker_id in enumerate(sorted_speaker_ids)}
        print(f"找到 {len(self.speaker_to_int)} 个说话人。")

        # 4. 填充最终样本列表
        self.samples = []
        for full_path, speaker_id, sr in temp_samples_metadata:
            if speaker_id in self.speaker_to_int:
                speaker_int = self.speaker_to_int[speaker_id]
                # 最终缓存的内容是：文件路径、整数标签、采样率
                self.samples.append((full_path, speaker_int, sr))

        # --- 缓存保存 ---
        print(f"样本列表填充完毕，总样本数: {len(self.samples)}。正在保存缓存...")
        cache_data = {
            'samples': self.samples, # 存储的是路径
            'speaker_to_int': self.speaker_to_int
        }
        with open(self.cache_path, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"缓存文件已保存到 {self.cache_path}。")


    def __getitem__(self, idx):
        """
        __getitem__ 现在负责从文件路径加载音频。
        """
        # 1. 从缓存中获取文件路径和元数据
        file_path, speaker_int, sr = self.samples[idx]
        
        # 2. **按需加载音频 (核心修改)**
        # 加载波形，而不是从内存中获取 Tensor
        waveform, original_sr = torchaudio.load(file_path)

        # 3. 后续处理 (重采样, 填充/截断, Mel-Spectrogram) 保持不变
        # ... (重采样逻辑)
        if original_sr != self.sample_rate:
            resampler = T.Resample(orig_freq=original_sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        # 填充/截断到固定长度
        # ... (填充/截断逻辑)
        if waveform.shape[1] < self.segment_length_samples:
            diff = self.segment_length_samples - waveform.shape[1]
            waveform = F.pad(waveform, (0, diff), "constant", 0)
        elif waveform.shape[1] > self.segment_length_samples:
            start = torch.randint(0, waveform.shape[1] - self.segment_length_samples + 1, (1,)).item()
            waveform = waveform[:, start:start + self.segment_length_samples]
            
        # 转换为 Mel-Spectrogram
        mel_spec = self.mel_spectrogram(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)
        
        return mel_spec_db.squeeze(0), torch.tensor(speaker_int).long()


    def __len__(self):
        return len(self.samples)

class LibriSpeechForMemoryBank(LibriSpeechSpeakerID):
    def __init__(self, augment=True, *args, **kwargs):
        """
        增加 augment 参数开关
        """
        super().__init__(*args, **kwargs)
        self.augment = augment
        # 初始化增强模块
        if self.augment:
            self.augmentor = WaveformAugmentations(sample_rate=self.sample_rate, augment_prob=0.5)

    def __getitem__(self, idx):
        # 1. 获取路径
        file_path, speaker_int, sr = self.samples[idx]
        
        # 2. 加载原始波形
        try:
            waveform, original_sr = torchaudio.load(file_path)
        except Exception as e:
            # 容错处理：如果文件损坏，随机返回另一个
            print(f"Error loading {file_path}: {e}")
            return self.__getitem__(random.randint(0, len(self.samples)-1))
        
        # 3. 基础重采样 (统一到 16k)
        if original_sr != self.sample_rate:
            resampler = T.Resample(orig_freq=original_sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)
        
        waveform = waveform.squeeze(0) # [T]

        # =========================================
        # 4. 数据增强 - 第一阶段 (改变长度的操作)
        # =========================================
        if self.augment:
            # 拉伸/变速 (这会改变 waveform 的长度 T)
            t0=time.time()
            # waveform = self.augmentor.time_stretch(waveform)
            t1=time.time()

        # =========================================
        # 5. 统一长度处理 (Padding / Cutting)
        # =========================================
        # 注意：必须在变速之后做，因为变速会改变时长
        current_len = waveform.shape[0]
        if current_len < self.segment_length_samples:
            diff = self.segment_length_samples - current_len
            # 随机 padding 位置 (左边 pad 多少，右边 pad 多少) 增加鲁棒性
            pad_left = random.randint(0, diff)
            pad_right = diff - pad_left
            waveform = F.pad(waveform.unsqueeze(0), (pad_left, pad_right), "constant", 0).squeeze(0)
        elif current_len > self.segment_length_samples:
            # 随机裁剪
            start = torch.randint(0, current_len - self.segment_length_samples + 1, (1,)).item()
            waveform = waveform[start:start + self.segment_length_samples]

        # =========================================
        # 6. 数据增强 - 第二阶段 (固定长度后的操作)
        # =========================================
        if self.augment:
            # 6.1 模拟压缩/有损编码
            t2=time.time()
            waveform = self.augmentor.simulate_compression(waveform)
            t3=time.time()
            # 6.2 模拟低音质 (重采样)
            waveform = self.augmentor.resample_quality_drop(waveform)
            t4=time.time()
            # 6.3 加噪声
            waveform = self.augmentor.add_noise(waveform)
            t5=time.time()
        # 7. 返回
        # wav2vec2 期望 [T] (DataLoader 会堆叠成 [B, T])
        return waveform
    
import torch
import torchaudio
import torchaudio.transforms as T
import random
import numpy as np

class RobustnessAugmentation:
    def __init__(self, sample_rate=16000, prob=0.5):
        self.sample_rate = sample_rate
        self.prob = prob
        
        # 1. 模拟低带宽/重采样 (Downsample -> Upsample)
        self.downsample_factors = [8000, 12000] 
        
        # 2. 变速/拉伸范围
        self.speed_range = (0.9, 1.1)

    def add_noise(self, waveform):
        """加入高斯白噪声"""
        if random.random() > self.prob:
            return waveform
            
        # 简单的 SNR 计算与加噪
        snr_db = random.uniform(10, 30)
        signal_power = waveform.pow(2).mean()
        if signal_power == 0: return waveform
        
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = torch.randn_like(waveform) * torch.sqrt(noise_power)
        return waveform + noise

    def simulate_compression(self, waveform):
        """模拟 Mu-law 压缩 (类似 G.711)"""
        if random.random() > self.prob:
            return waveform, False # False 表示未压缩
            
        # Mu-law encoding/decoding 模拟量化噪声
        encoded = torchaudio.functional.mu_law_encoding(waveform, quantization_channels=256)
        decoded = torchaudio.functional.mu_law_decoding(encoded, quantization_channels=256)
        return decoded, True # True 表示已压缩

    def resample_quality_drop(self, waveform):
        """模拟采样率降低"""
        if random.random() > self.prob:
            return waveform
            
        target_sr = random.choice(self.downsample_factors)
        resampler_down = T.Resample(self.sample_rate, target_sr, dtype=waveform.dtype)
        resampler_up = T.Resample(target_sr, self.sample_rate, dtype=waveform.dtype)
        return resampler_up(resampler_down(waveform))

    def time_stretch(self, waveform):
        """变速 (改变时长)"""
        if random.random() > self.prob:
            return waveform, 1.0
            
        speed = random.uniform(self.speed_range[0], self.speed_range[1])
        # 使用 Resample 模拟变速效果 (Pitch Shift + Time Stretch)
        resampler = T.Resample(
            orig_freq=int(self.sample_rate / speed), 
            new_freq=self.sample_rate, 
            dtype=waveform.dtype
        )
        
        # 处理维度 [C, T]
        if waveform.ndim == 2:
            out = resampler(waveform)
        else:
            out = resampler(waveform.unsqueeze(0)).squeeze(0)
            
        return out, speed

    def __call__(self, waveform, metadata=None):
        """
        WaveDataset 会调用这个方法。
        注意：WaveDataset 支持 numpy 和 tensor，我们需要统一处理。
        """
        # 1. 统一转为 Tensor 处理
        is_numpy = False
        if isinstance(waveform, np.ndarray):
            is_numpy = True
            waveform = torch.from_numpy(waveform)
            # 确保是 float
            if not waveform.is_floating_point():
                waveform = waveform.float()

        # 确保维度是 [C, T] (torchaudio 标准)
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)

        # ------------------------------------------------
        # 2. 应用增强
        # ------------------------------------------------
        
        # # A. 变速 (会改变长度)
        # waveform, speed = self.time_stretch(waveform)
        # if metadata is not None and speed != 1.0:
        #     # 可以在这里更新 metadata，比如记录速度变化
        #     # 注意：您的 Dataset 默认 speed_label=5，您可能需要定义映射逻辑
        #     metadata["speed_label"] = speed # 或者映射后的整数类别

        # B. 压缩模拟
        waveform, is_compressed = self.simulate_compression(waveform)
        if metadata is not None and is_compressed:
            metadata["compression_label"] = 1 # 假设 1 代表有压缩

        # C. 降采样模拟 (带宽限制)
        waveform = self.resample_quality_drop(waveform)

        # D. 加噪声
        waveform = self.add_noise(waveform)

        # ------------------------------------------------
        # 3. 还原格式
        # ------------------------------------------------
        if is_numpy:
            waveform = waveform.numpy()
            
        return waveform