import torch
import torch.nn as nn
import torchaudio
import torchaudio
from controlled_ex.Tinyvit.tiny_vit import tiny_vit_21m_224

import torch.nn.functional as F

from models.MultiView.resnet.resnet import get_model
from myutils.zyz.my_utils import post_process

class ResNet(nn.Module):
    def __init__(self, verbose=0, pretrained=True):
        super().__init__()

        self.model = get_model(pretrained=pretrained).encoder

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

        if first_conv:
            x = self.model.conv1(x)
            x = self.model.bn1(x)
            x = self.model.relu(x)
            x = self.model.maxpool(x)
        if self.verbose:
            print("ResNet Stage 1: input after first conv shape", x.shape)

        x = self.model.layer1(x)
        if self.verbose:
            print("ResNet Stage 1: output shape", x.shape)
        return x, raw_spec

    def compute_rest_stage(self, x):
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)  # (64, 512, 9, 9) -> (64, 512)
        x = x.reshape(x.size(0), -1)
        x = self.feature_norm(x)
        return x
    
    def feature_norm(self, code):
        code_norm = code.norm(p=2, dim=1, keepdim=True) / 10
        code = torch.div(code, code_norm)
        return code