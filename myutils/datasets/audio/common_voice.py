# import os
# import glob
# from datasets import load_dataset, Audio, concatenate_datasets

# # --- 配置参数 ---
# # 替换为您的 Common Voice 根目录
# COMMON_VOICE_ROOT = "/home/zyz/data/common_voice" 
# # 替换为您希望保存处理后数据集的目录
# PROCESSED_DATA_ROOT = "/home/zyz/data/common_voice_processed_16k" 
# TARGET_SR = 16000 # 目标采样率

# # 明确指定需要处理的语言列表
# # 请根据您的本地文件结构调整此列表
# LANGUAGES_TO_PROCESS = ['en', 'uk', 'fr', 'ru','it','pl'] 

# # 确保目标保存目录存在
# os.makedirs(PROCESSED_DATA_ROOT, exist_ok=True)


# def process_and_save_language(lang_code):
#     """
#     加载指定语言的原始 Parquet 文件，重采样后保存为优化的 HF 格式。
#     """
    
#     print(f"\n--- 开始处理语言: {lang_code} ---")
#     lang_dir = os.path.join(COMMON_VOICE_ROOT, lang_code)
    
#     # 1. 查找 Parquet 文件
#     # 使用递归查找 (**) 确保找到所有子文件夹中的 parquet 文件 (如 train/validation/test)
#     parquet_files = glob.glob(os.path.join(lang_dir, "**", "*.parquet"), recursive=True)

#     if not parquet_files:
#         print(f"❌ 警告: 在 {lang_dir} 下未找到任何 .parquet 文件，跳过此语言。")
#         return
        
#     print(f"找到 {len(parquet_files)} 个 Parquet 文件，正在进行首次加载...")

#     # 2. 首次加载 (此步骤较慢，但只执行一次)
#     try:
#         # 使用 load_dataset 加载所有文件，并定义为一个 'train' 分割
#         # 返回的是一个 DatasetDict，我们提取其中的 'train'
#         cv_dataset_dict = load_dataset(
#             "parquet", 
#             data_files={'train': parquet_files}
#         )
#         raw_dataset = cv_dataset_dict['train']
#         print(f"✅ 原始加载完成，共 {len(raw_dataset)} 条记录。")
#     except Exception as e:
#         print(f"❌ 加载 {lang_code} 数据集时发生错误: {e}，跳过此语言。")
#         return

#     # 3. 数据预处理：重采样
#     print(f"正在将音频重采样到 {TARGET_SR} Hz...")
#     try:
#         raw_dataset = raw_dataset.cast_column(
#             "audio", 
#             Audio(sampling_rate=TARGET_SR)
#         )
#         print("✅ 重采样完成。")
#     except Exception as e:
#         print(f"❌ 重采样 {lang_code} 数据集时发生错误: {e}，跳过此语言。")
#         return

#     # 4. 保存为优化的本地格式 (生成 dataset_info.json)
#     save_path = os.path.join(PROCESSED_DATA_ROOT, lang_code)
#     print(f"正在将数据集保存到优化目录: {save_path}")
    
#     raw_dataset.save_to_disk(save_path)
    
#     print(f"🎉 语言 {lang_code} 数据集已成功保存！")


# # --- 运行批量处理 ---
# for lang in LANGUAGES_TO_PROCESS:
#     process_and_save_language(lang)

# print("\n=== 所有指定语言处理完成！===")


import io
import os
import subprocess
import tempfile
import torch
import torchaudio
import torchaudio.transforms as T
from datasets import load_from_disk, concatenate_datasets
from torch.utils.data import Dataset
from typing import Any, Dict, List, Optional, Sequence, Union

from myutils.datasets.audio.Augment import ASVspoofDomainAugmentation, PseudoAnomalyAugmentor
from myutils.datasets.audio._chime import cut_length
CODEC_PRESETS = [
    {
        "fmt": "mp3",
        "encoder": None,
        "bit_rates": [32000, 64000, 96000, 128000],
        "qscales": [2, 4, 6, 8]
    },
    {
        "fmt": "ogg",
        "encoder": "vorbis",
        "bit_rates": [48000, 64000, 96000],
        "qscales": [3, 5, 7]
    },
    {
        "fmt": "flac",
        "encoder": None,
        "bit_rates": [None],  # FLAC 通常不用比特率控制
        "qscales": [0, 3, 6, 8]
    }
]
# --- 音频处理的固定参数 (用于生成音频张量) ---
TARGET_SR = 16000 # 目标采样率
DURATION = 3.0    # 音频切片时长（秒）
N_MELS = 64       # Mel 频带数
HOP_LENGTH = 512  

class Common_voice_dataset(Dataset):
    """
    一个通用的 PyTorch Dataset 类，用于加载和合并多语言 Common Voice 数据集，
    并返回 Mel-Spectrogram 张量和对应的语言标签。
    """

    def __init__(self, processed_root_path: str, aug = False, asv_aug = False, pseudo_aug = False, stage='stage1', lang_codes: Union[str, List[str]] = 'all'):
        """
        Args:
            processed_root_path: 存储着所有语言子文件夹的根目录。
            lang_codes: 要加载的语言代码列表，例如 ['en', 'uk'] 或 'all'。
        """
        self.processed_root = processed_root_path
        self.stage = stage
        # 1. 确定要加载的语言列表
        if lang_codes == 'all':
            all_dirs = [d for d in os.listdir(self.processed_root) 
                        if os.path.isdir(os.path.join(self.processed_root, d))]
            self.target_langs = sorted(all_dirs)
        elif isinstance(lang_codes, str):
            self.target_langs = [lang_codes]
        else:
            self.target_langs = lang_codes

        # 2. 加载和拼接所有语言数据集
        self.hf_dataset = self._load_and_concatenate_datasets()
        
        # 3. 初始化音频处理参数和变换器
        self.target_length = int(TARGET_SR * DURATION)
        self.mel_spectrogram_transform = T.MelSpectrogram(
            sample_rate=TARGET_SR,
            n_fft=1024,
            hop_length=HOP_LENGTH,
            n_mels=N_MELS
        )
        self.amplitude_to_db = T.AmplitudeToDB()
        self.aug = aug
        self.asv_aug = asv_aug
        self.pseudo_aug = pseudo_aug
        self.domain_augmenter = ASVspoofDomainAugmentation(orig_freq=TARGET_SR)
        self.pseudo_augmenter = PseudoAnomalyAugmentor(sample_rate=TARGET_SR)
        self.augmentations = [
            add_noise,
            lambda w, sr=TARGET_SR: random_rate_compression(w, sr), # 换码率压缩
            random_gain,
            # lambda w, sr=TARGET_SR: random_pitch_shift(w, sr),
        ]

    def __len__(self):
        """返回总数据集大小。"""
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        """
        返回：(音频张量, 语言标签)
        音频张量 (Tensor) 是 3 通道 Mel-Spectrogram。
        """
        # 1. 从 Hugging Face Dataset 中获取数据
        example = self.hf_dataset[idx]
        label = 1.0
        # 获取波形
        audio_array = example['audio']['array']
        waveform = torch.from_numpy(audio_array).float().unsqueeze(0) # (1, Length)
        # ==========================================
        # Stage 2 逻辑: 制造 Pseudo Anomaly (负样本)
        # ==========================================
        is_pseudo_anomaly = False
        if self.pseudo_aug:
            # if self.stage == 'stage2':
                # 50% 概率变成伪异常
            if random.random() < 0.5:
                waveform = self.pseudo_augmenter(waveform)
                label = 0.0 # 标记为 Fake (Anomaly)
                is_pseudo_anomaly = True
            # ==========================================
        # Stage 1 & Stage 2 (Real部分) 逻辑: 域泛化增强
        # ==========================================
        # 只有当它是 Real (label=1) 时，我们才做 ASVspoof 模拟增强
        # 如果它已经是 Pseudo Anomaly 了，就没必要再模拟电话信道了（或者也可以模拟，看你策略，通常保持纯粹破坏即可）
        if not is_pseudo_anomaly and self.asv_aug:
            if random.random() < 0.5:
                waveform = self.domain_augmenter(waveform)
        if not is_pseudo_anomaly and self.aug:
            # 1. 域增强 (Label 保持 1)
            
            # 随机应用部分或全部增强
            random.shuffle(self.augmentations)
            # 随机选择 1 到所有增强中的一个子集进行应用
            num_augs_to_apply = random.randint(1, len(self.augmentations)) 
            
            for i in range(num_augs_to_apply):
                aug_func = self.augmentations[i]
                # 注意: 增强函数内部需要处理 unsqueeze(0) 和 squeeze(0) 的问题
                # 这里统一传 (1, Length) 进去，期望返回 (Length,)
                augmented_waveform = aug_func(augmented_waveform.unsqueeze(0) if augmented_waveform.dim()==1 else augmented_waveform)

        waveform=waveform.unsqueeze(0) if waveform.dim()==1 else waveform
        waveform = cut_length(waveform=waveform,target_length=self.target_length)
        # 严格按照要求返回：(音频张量, 语言标签)
        return {"audio":waveform.squeeze(0).detach(), "label":label,"domain_label":"common_voice"} 
    def _load_and_concatenate_datasets(self):
        """加载所有指定的语言数据集并拼接成一个 Hugging Face Dataset。"""
        
        all_datasets = []
        print("--- 正在加载并拼接数据集 ---")
        
        for lang_code in self.target_langs:
            lang_path = os.path.join(self.processed_root, lang_code)
            
            if not os.path.exists(lang_path):
                print(f"❌ 警告: 未找到语言 {lang_code} 的已处理数据集路径，跳过。")
                continue

            try:
                hf_dataset = load_from_disk(lang_path)
                
                # 只添加语言标签 (lang_tag)，不添加任何 DFM 标签
                hf_dataset = hf_dataset.map(
                    lambda x: {'lang_tag': [lang_code] * len(x['path'])}, # 使用 x['path'] 来获取批次大小
                    batched=True, 
                )
                all_datasets.append(hf_dataset)
                print(f"✅ 成功加载 {lang_code}，大小: {len(hf_dataset)}")
                
            except Exception as e:
                print(f"❌ 加载或处理语言 {lang_code} 时出错: {e}")

        if not all_datasets:
            raise RuntimeError("未成功加载任何数据集，请检查路径和语言代码。")
            
        full_dataset = concatenate_datasets(all_datasets)
        print(f"🎉 所有语言拼接完成，总记录数: {len(full_dataset)}")
        return full_dataset
# # --- 使用示例 ---
# if __name__ == "__main__":
    
#     # 替换成您保存所有已处理语言数据集的根目录
#     processed_root_dir = "/home/zyz/data/common_voice_processed_16k" 
    
#     try:
#         # 实例化：加载所有语言并合并
#         full_cv_dataset = Common_voice_dataset(processed_root_dir, lang_codes='all')
#         print(f"\n数据集成功创建，总样本数: {len(full_cv_dataset)}")

#         # 创建 PyTorch DataLoader
#         from torch.utils.data import DataLoader
#         train_loader = DataLoader(full_cv_dataset, batch_size=64, shuffle=False, num_workers=4)

#         # 遍历第一个 Batch
#         for specs, lang_tags in train_loader:
#             print(f"\n第一个 Batch:")
#             print(f"  音频张量 (Specs) 形状: {specs.shape}") # [64, 3, 64, ~94]
#             print(f"  前五个样本的语言标签: {lang_tags[:5]}")
#             break
            
#     except RuntimeError as e:
#         print(f"\n创建数据集失败: {e}")

import torch
import torchaudio.transforms as T
import torchaudio.functional as F
import random
import os
from typing import Union, List
from torch.utils.data import Dataset
from datasets import load_from_disk, concatenate_datasets


# ------------------------------------------------------------
# 辅助数据增强函数 (在 __getitem__ 中调用)
# ------------------------------------------------------------

def add_noise(waveform: torch.Tensor, snr_range: tuple = (5.0, 30.0)) -> torch.Tensor:
    """
    加高斯白噪声。
    Args:
        waveform: 输入波形 (1, Length)。
        snr_range: 信噪比 (SNR) 范围 (dB)。
    Returns:
        加噪后的波形。
    """
    if random.random() < 0.5: # 50% 的概率应用
        snr_db = random.uniform(snr_range[0], snr_range[1])
        noise = torch.randn_like(waveform)
        # 确保波形是二维的 (1, Length)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
            
        # 计算当前信号功率（均方值）
        signal_power = torch.mean(waveform**2)
        # 根据 SNR 计算所需的噪声功率
        noise_power = signal_power / (10**(snr_db / 10.0))
        # 缩放噪声
        noise_scale = torch.sqrt(noise_power / torch.mean(noise**2))
        
        return (waveform + noise * noise_scale).squeeze(0)
    return waveform.squeeze(0)
def ffmpeg_compress_pipe(waveform: torch.Tensor, sample_rate: int,
                        format: str = "mp3", bitrate: int = 64000) -> torch.Tensor:
    """
    完全内存操作：使用 FFmpeg 管道功能，不生成任何文件
    
    Args:
        waveform: (C, T) 或 (T,) 的张量
        sample_rate: 采样率
        format: 压缩格式 ('mp3', 'aac', 'opus', 'ogg')
        bitrate: 比特率（bps）
    
    Returns:
        压缩后的波形，保持原始形状
    """
    # 确保正确的形状 (C, T)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # (1, T)
    
    channels = waveform.shape[0]
    
    # 1. 将音频转换为原始 PCM 字节流
    # WAV 头 + PCM 数据
    import struct
    buffer = io.BytesIO()
    
    # 写入 WAV 头（FFmpeg 可以从原始 PCM 读取，但 WAV 头更可靠）
    torchaudio.save(buffer, waveform, sample_rate, format="wav")
    wav_bytes = buffer.getvalue()
    
    # 2. 构建 FFmpeg 管道命令
    # 编码阶段：从 stdin 读取 -> 压缩编码 -> 输出到 stdout
    encode_cmd = [
        "ffmpeg",
        "-i", "pipe:0",           # 从 stdin 读取输入
        "-f", format,            # 输出格式
        "-c:a", get_codec(format),  # 编码器
        "-b:a", f"{bitrate}",    # 比特率
        "-ar", str(sample_rate), # 保持采样率
        "-ac", str(channels),    # 保持声道数
        "pipe:1",                # 输出到 stdout
        "-loglevel", "error",    # 只显示错误
        "-y"                     # 覆盖输出
    ]
    
    # 解码阶段：从 stdin 读取压缩数据 -> 解码为 WAV -> 输出到 stdout
    decode_cmd = [
        "ffmpeg",
        "-i", "pipe:0",           # 从 stdin 读取压缩数据
        "-f", "wav",             # 输出为 WAV
        "-c:a", "pcm_s16le",     # PCM 编码
        "pipe:1",                # 输出到 stdout
        "-loglevel", "error",
        "-y"
    ]
    
    try:
        # 3. 执行编码管道
        encode_proc = subprocess.Popen(
            encode_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        encoded_bytes, encode_err = encode_proc.communicate(input=wav_bytes)
        
        if encode_proc.returncode != 0:
            raise RuntimeError(f"编码失败: {encode_err.decode()}")
        
        # 4. 执行解码管道
        decode_proc = subprocess.Popen(
            decode_cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        
        decoded_bytes, decode_err = decode_proc.communicate(input=encoded_bytes)
        
        if decode_proc.returncode != 0:
            raise RuntimeError(f"解码失败: {decode_err.decode()}")
        
        # 5. 从内存加载音频
        buffer = io.BytesIO(decoded_bytes)
        compressed, sr = torchaudio.load(buffer)
        
        # 确保采样率一致
        if sr != sample_rate:
            compressed = torchaudio.functional.resample(compressed, sr, sample_rate)
        
        # 保持原始形状
        if waveform.shape[0] == 1:
            return compressed.squeeze(0)
        return compressed
        
    except Exception as e:
        print(f"FFmpeg 管道处理失败: {e}")
        return waveform  # 失败时返回原始音频

def get_codec(format: str) -> str:
    """获取格式对应的编码器"""
    codec_map = {
        "mp3": "libmp3lame",
        "aac": "aac",
        "opus": "libopus",
        "ogg": "libvorbis",
        "flac": "flac",
        "wav": "pcm_s16le",
        "m4a": "aac"
    }
    return codec_map.get(format, "libmp3lame")
def compress_with_audioeffector_replacement(
    waveform: torch.Tensor,
    sample_rate: int,
    *,
    fmt: str = "mp3",
    encoder: Optional[str] = None,
    bit_rate: Optional[int] = None,
    qscale: Optional[int] = None,
    pad_end: bool = True,
) -> torch.Tensor:
    """
    替换原来的 AudioEffector，保持相同接口
    """
    # 设置默认参数
    if bit_rate is None:
        if fmt == "mp3":
            bit_rate = 64000
        elif fmt == "opus":
            bit_rate = 32000
        elif fmt == "aac":
            bit_rate = 48000
        else:
            bit_rate = 64000
    
    # 如果有 qscale，转换为比特率（简化映射）
    if qscale is not None and bit_rate is None:
        # qscale 0-9 映射到比特率
        bit_rate = int(32000 + (10 - qscale) * 10000)
    
    # 使用管道处理
    result = ffmpeg_compress_pipe(
        waveform, sample_rate,
        format=fmt,
        bitrate=bit_rate or 64000
    )
    
    # 处理填充逻辑
    if pad_end and result.shape[-1] < waveform.shape[-1]:
        pad_len = waveform.shape[-1] - result.shape[-1]
        result = torch.nn.functional.pad(result, (0, pad_len))
    
    return result

# 保持原有的 random_rate_compression 接口不变
def random_rate_compression(
    waveform: torch.Tensor,
    sample_rate: int,
    *,
    presets: Sequence[Dict[str, Any]] = CODEC_PRESETS,
    p: float = 0.5,
) -> torch.Tensor:
    """
    使用 FFmpeg 管道的随机压缩增强
    """
    if random.random() > p or not presets:
        return waveform
    
    preset = random.choice(presets)
    
    bit_rate = None
    qscale = None
    
    bit_rates = preset.get("bit_rates") or []
    qscales = preset.get("qscales") or []
    
    # 优先随机选 bit_rate，没有的话用 qscale
    if bit_rates and random.random() < 0.8:
        bit_rate = random.choice(bit_rates)
    elif qscales:
        qscale = random.choice(qscales)
    elif bit_rates:
        bit_rate = random.choice(bit_rates)
    
    return compress_with_audioeffector_replacement(
        waveform,
        sample_rate=sample_rate,
        fmt=preset["fmt"],
        encoder=preset.get("encoder"),
        bit_rate=bit_rate,
        qscale=qscale,
    )


def random_gain(waveform: torch.Tensor, gain_range: tuple = (-6.0, 6.0)) -> torch.Tensor:
    """
    随机调整音量 (增益/衰减)。
    Args:
        waveform: 输入波形 (1, Length)。
        gain_range: 增益范围 (dB)。
    Returns:
        调整音量后的波形。
    """
    if random.random() < 0.6: # 60% 的概率应用
        gain_db = random.uniform(gain_range[0], gain_range[1])
        # 使用 torchaudio 的 gain 函数
        # 注意: gain 期望输入是浮点数，您的 waveform 已经是浮点数
        return F.gain(waveform, gain_db=gain_db).squeeze(0)
    return waveform.squeeze(0)

def random_pitch_shift(waveform: torch.Tensor, sample_rate: int, n_steps_range: tuple = (-2, 2)) -> torch.Tensor:
    """
    随机调整声高 (音高/Pitch Shift)。
    
    Args:
        waveform: 输入波形 (1, Length)。
        sample_rate: 音频的采样率 (例如 16000)。
        n_steps_range: 音高调整的半音 (semitones) 范围。
        
    Returns:
        调整声高后的波形。
    """
    N_FFT=1024
    if random.random() < 0.4:
        n_steps = random.randint(n_steps_range[0], n_steps_range[1])
        
        input_waveform = waveform.unsqueeze(0) if waveform.dim() == 1 else waveform
        
        # 💥 最终修正：将 sample_rate 作为关键字参数传入
        return F.pitch_shift(
            input_waveform,
            sample_rate=sample_rate, # <-- 修正：作为必需的关键字参数传入
            n_steps=n_steps,
            n_fft=N_FFT,       
            hop_length=HOP_LENGTH 
        ).squeeze(0)

    return waveform.squeeze(0)