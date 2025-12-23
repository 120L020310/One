import torch
from Oneclass_XLSR_lit_wepe import get_perturbed_state_dict
from models.SLSforASVspoof.teacher import XLSR_Teacher
from torch.func import functional_call
import torch.nn as nn
import torch.nn.functional as F
# model = XLSR_Teacher(proj=True)
# print(model)
# 测试专用 Augmentor：模拟电话信道/压缩
class TestTimeAugmentor(nn.Module):
    def forward(self, audio):
        # 1. 降采样到 8k 再升回来 (模拟极其恶劣的电话线)
        # 训练时模型没见过这个，Real 应该依然很稳，Fake 的高频伪影会丢失
        orig_len = audio.shape[-1]
        down = F.interpolate(audio.unsqueeze(1), scale_factor=0.25, mode='linear') # 48k -> 12k
        up = F.interpolate(down, size=orig_len, mode='linear').squeeze(1)
        return up
    
def get_wepe(model, audio, noise_ratio=0.1,mode="multiplicative"):
    # 带噪的前向传播
        noisy_state_dict = get_perturbed_state_dict(
            model, 
            noise_ratio=noise_ratio,  # 建议设大一点，比如 0.1 或 0.15
            target_layer_indices=range(12, 24), # 攻击后 12 层
            mode=mode
        )
            
        # 使用 functional_call 执行带噪推理
        res_noisy = functional_call(model, noisy_state_dict, args=(audio,), kwargs={})
        feat_noisy = res_noisy["final_feat"]
        return feat_noisy

def get_augment(audio, augmentor,model):
    audio_noisy_input = augmentor(audio)
    
    # 将增强后的音频送入同一个模型 (共享权重)
    res_noisy = model(audio_noisy_input)
    feat_noisy = res_noisy["final_feat"]      
    return feat_noisy 

def get_combined_noisy_feat(model, audio, augmentor, noise_ratio=0.1, mode="multiplicative"):
    """
    同时应用输入端增强 (Augmentor) 和 模型参数端增强 (WEPE)
    """
    # 1. 输入端扰动：先对原始音频进行数据增强
    # audio_noisy_input shape: (Batch, Length)
    audio_noisy_input = augmentor(audio)
    
    # 2. 参数端扰动：获取 WEPE 扰动后的 state_dict
    noisy_state_dict = get_perturbed_state_dict(
        model, 
        noise_ratio=noise_ratio,
        target_layer_indices=range(12, 24), 
        mode=mode
    )
    
    # 3. 组合推理：使用带噪参数对带噪输入进行前向传播
    # 使用 functional_call 避免直接修改 model 的权重
    res_noisy = functional_call(
        model, 
        noisy_state_dict, 
        args=(audio_noisy_input,), 
        kwargs={}
    )
    
    return res_noisy["final_feat"]

def choose_noisy(self,audio):
    strategy_idx = torch.randint(0, 3, (1,)).item()
    strategy_idx = 2
    # --- 分支逻辑 ---
    if strategy_idx == 0:
        # 1. 只用 WEPE
        feat_noisy = get_wepe(self.model, audio, noise_ratio=0.1, mode="multiplicative")
        
    elif strategy_idx == 1:
        # 2. 只用 Augment 1 (SafeRaw)
        feat_noisy = get_augment(audio, self.augmentor1, self.model)
        
    # elif strategy_idx == 2:
    #     # 3. 只用 Augment 2 (TransformerHard)
    #     feat_noisy = get_augment(audio, self.augmentor2, self.model)
        
    elif strategy_idx == 2:
        # 4. 用 WEPE + Augment 1
        feat_noisy = get_combined_noisy_feat(
            self.model, audio, self.augmentor1, noise_ratio=0.1, mode="multiplicative"
        )
        
    # elif strategy_idx == 4:
    #     # 5. 用 WEPE + Augment 2
    #     feat_noisy = get_combined_noisy_feat(
    #         self.model, audio, self.augmentor2, noise_ratio=0.1, mode="multiplicative"
    #     )
    return feat_noisy