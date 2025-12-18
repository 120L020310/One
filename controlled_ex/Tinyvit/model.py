import torch
import torch.nn as nn
import torchaudio
import torchaudio
from controlled_ex.Tinyvit.tiny_vit import tiny_vit_21m_224

import torch.nn.functional as F
from myutils.torchaudio.transforms import SpecAugmentBatchTransform
from myutils.zyz.my_utils import post_process


class AddGaussianNoise(nn.Module):
    def __init__(self, mean=0., std=0.1):
        super().__init__()
        self.mean = mean
        self.std = std
        
    def forward(self, spectrogram):
        # 只在训练阶段添加噪声
        if self.training:
            # 创建与频谱图相同形状的噪声
            noise = torch.randn_like(spectrogram) * self.std + self.mean
            # 添加噪声并确保值在合理范围内
            noisy_spec = torch.clamp(spectrogram + noise, min=0, max=1)
            return noisy_spec
        return spectrogram
    

class TinyVit(nn.Module):
    def __init__(self, verbose=0, pretrained=True):
        super().__init__()

        self.model = tiny_vit_21m_224(pretrained=True)
        self.post_extractor = post_process(hidden_dim=3072)
        # from torchvision.models import resnet18
        # model = resnet18(weights='DEFAULT')
        # sd = model.state_dict()
        # sd['conv1.weight'] = torch.mean(sd['conv1.weight'], dim=1, keepdims=True)
        # _ = self.model.load_state_dict(sd, strict=False)

        self.spectrogram = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=187)
        # self.spectrogram = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=353) # original
        self.verbose = verbose
        self.noise_adder = AddGaussianNoise(std=0.05)
        self.spec_aug = SpecAugmentBatchTransform.from_policy("ss")
        

    def preprocess(self, x, stage="test"):
        # x = self.model.spectrogram(x)
        x = self.spectrogram(x)
        if stage == "train":
             x = self.noise_adder(x)
        x = F.interpolate(x, size=(224, 224), mode="bilinear")
        x = torch.log(x + 1e-7)
        # x = (x - torch.mean(x)) / (torch.std(x) + 1e-9)

        x = (x - torch.mean(x, dim=(1, 2, 3), keepdim=True)) / (
            torch.std(x, dim=(1, 2, 3), keepdim=True) + 1e-9
        )
        if stage =="train":
            if self.spec_aug is not None:
                # print('use spec aug', x.shape)
                x = self.spec_aug.batch_apply(x)
        return x

    def compute_stage1(self, x, stage="test",preprocess=True, first_conv=True, spec_aug=None):
        if preprocess:
            x = self.preprocess(x,stage)
            raw_spec = x
            # if self.spec_aug is not None:
            #     # print('use spec aug', x.shape)
            #     y = self.spec_aug.batch_apply(x)
                # z = torch.sum(torch.abs(y-x))
        x = torch.cat([x, x, x], dim=1)
        x = self.model.patch_embed(x)

        return x, raw_spec

    def compute_rest_stage(self, x):
        x = self.model.layers[0](x)
        x = self.model.layers[1](x)
        x = self.model.layers[2](x)
        x = self.model.layers[3](x)
        x = self.model.norm_head(x)
        # x = self.post_extractor(x)
        # x = x.mean(1)
        # x = self.feature_norm(x)
        return x


    def feature_norm(self, code):
        code_norm = code.norm(p=2, dim=1, keepdim=True) / 10
        code = torch.div(code, code_norm)
        return code



