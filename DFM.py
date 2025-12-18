from models.MultiView.utils import WavLM_1D
from myutils.torch.deepfake_detection.audio import DeepfakeAudioClassification

import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

class EmotionModel(Wav2Vec2PreTrainedModel):
    r"""Speech emotion classifier."""

    def __init__(self, config):

        super().__init__(config)

        self.config = config
        self.wav2vec2 = Wav2Vec2Model(config)
        self.init_weights()

    def forward(
            self,
            input_values,
    ):

        outputs = self.wav2vec2(input_values,output_hidden_states=True)
        
        # logits = self.classifier(hidden_states)

        return outputs
    
import torch
import torch.nn as nn
from models.WaveLM.wavlm import BaseLine as WavLM
class FeatureExtractor(nn.Module):
    def __init__(self, target_layer=6):
        """
        original_model: 您那个 test_lit.model (EmotionModel)
        target_layer: DFM 建议取中间层，wav2vec2-large 有 24 层，建议取 6-12 层之间
        """
        super().__init__()
        # model_name = '/home/zyz/data/work2/wav2vec2-large-robust-12-ft-emotion-msp-dim'
        # self.model = EmotionModel.from_pretrained(model_name)
        self.model = WavLM_1D()
        self.target_layer = target_layer
        self.model.eval() # 强制 eval 模式

    def forward(self, x):
        # 假设 x 是 [batch, seq_len]
        with torch.no_grad():
            features = self.model(x.squeeze(1),self.target_layer)
            
        return features # Shape: [batch, frames, dim]
    

# tensor = torch.randn([64,1,48000])
# print(tensor)
# fe=FeatureExtractor()
# print(fe(tensor).shape)