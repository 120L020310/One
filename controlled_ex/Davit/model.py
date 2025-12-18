import timm
import torch
import torch.nn as nn
import torchaudio
import torchaudio
from controlled_ex.Tinyvit.tiny_vit import tiny_vit_21m_224

import torch.nn.functional as F

from myutils.zyz.my_utils import post_process

class Davit(nn.Module):
    def __init__(self, verbose=0, pretrained=True):
        super().__init__()

        self.model = timm.create_model('davit_base.msft_in1k', pretrained=True, 
                          pretrained_cfg_overlay=dict(file='/home/zyz/.cache/huggingface/hub/davit_base/pytorch_model.bin'))
        self.post_extractor = post_process(hidden_dim=4096)
        # from torchvision.models import resnet18
        # model = resnet18(weights='DEFAULT')
        # sd = model.state_dict()
        # sd['conv1.weight'] = torch.mean(sd['conv1.weight'], dim=1, keepdims=True)
        # _ = self.model.load_state_dict(sd, strict=False)

        self.spectrogram = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=187)
        # self.spectrogram = torchaudio.transforms.Spectrogram(n_fft=512, hop_length=353) # original

        self.verbose = verbose
        self.fc = nn.Linear(in_features=1024, out_features=512,bias=True)

    def preprocess(self, x, stage="test"):
        # x = self.model.spectrogram(x)
        x = self.spectrogram(x)
        x = F.interpolate(x, size=(224, 224), mode="bilinear")
        x = torch.log(x + 1e-7)
        # x = (x - torch.mean(x)) / (torch.std(x) + 1e-9)

        x = (x - torch.mean(x, dim=(1, 2, 3), keepdim=True)) / (
            torch.std(x, dim=(1, 2, 3), keepdim=True) + 1e-9
        )
        return x

    def compute_stage1(self, x, preprocess=True, first_conv=True, spec_aug=None):
        if preprocess:
            x = self.preprocess(x)
            raw_spec = x
            if spec_aug is not None:
                # print('use spec aug', x.shape)
                x = spec_aug.batch_apply(x)
        x = torch.cat([x, x, x], dim=1)
        x = self.model.stem(x)

        return x, raw_spec

    def compute_rest_stage(self, x):
        x = self.model.stages[0](x)
        x = self.model.stages[1](x)
        x = self.model.stages[2](x)
        x = self.model.stages[3](x)
        x = self.model.norm_pre(x)
        x = self.model.head.global_pool(x)
        x = self.model.head.norm(x)
        x = self.model.head.flatten(x)
        x = self.fc(x)
        # x = x.permute(0, 2, 3, 1).contiguous().view(-1, 49, 768)
        # x = self.post_extractor(x)
        # x = x.mean(1)
        # x = self.feature_norm(x)
        return x


    def feature_norm(self, code):
        code_norm = code.norm(p=2, dim=1, keepdim=True) / 10
        code = torch.div(code, code_norm)
        return code



