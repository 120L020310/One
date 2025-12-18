import torch
import torch.nn as nn
import torchaudio
import torchaudio
from controlled_ex.Tinyvit.tiny_vit import tiny_vit_21m_224

import torch.nn.functional as F

from myutils.zyz.my_utils import post_process

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
        x = self.model.patch_embed(x)

        return x, raw_spec

    def compute_rest_stage(self, x):
        x = self.model.layers[0](x)
        x = self.model.layers[1](x)
        x = self.model.layers[2](x)
        x = self.model.layers[3](x)
        x = self.model.norm_head(x)
        x = self.post_extractor(x)
        # x = x.mean(1)
        # x = self.feature_norm(x)
        return x


    def feature_norm(self, code):
        code_norm = code.norm(p=2, dim=1, keepdim=True) / 10
        code = torch.div(code, code_norm)
        return code



