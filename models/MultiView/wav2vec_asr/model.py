from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
import torch
import torch.nn as nn
from torch.nn import LayerNorm
from collections import OrderedDict
import torch.nn.functional as F
import torch.nn as nn
# processor = AutoProcessor.from_pretrained("/home/zyz/data/asr_whisper")
# model = AutoModelForSpeechSeq2Seq.from_pretrained("/home/zyz/data/asr_whisper")
# print(model)
# import datasets
# ds = datasets.load_dataset('/home/zyz/data/salt', 'default', split='test')
# ds0 = ds[0]
# audio = ds[0]['audio']['array']
# sample_rate = ds[0]['audio']['sampling_rate']

# input_features = processor(
#             audio, sampling_rate=sample_rate, return_tensors="pt").input_features
from transformers import AutoProcessor, AutoModelForCTC

from myutils.zyz.my_utils import post_process
model = AutoModelForCTC.from_pretrained("/home/zyz/data/openslr")

class openslr_ASR(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.model = AutoModelForCTC.from_pretrained("/home/zyz/data/openslr")
        self.post_extractor = post_process(hidden_dim=16709)
    def compute_stage1(self,x):
        x = x.squeeze(1)
        extract_features = self.model.wav2vec2.feature_extractor(x)
        extract_features = extract_features.transpose(1, 2)
        hidden_states,_ = self.model.wav2vec2.feature_projection(extract_features)
        position_embeddings = self.model.wav2vec2.encoder.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.model.wav2vec2.encoder.dropout(hidden_states)
        # hidden_states = self.model(x.squeeze(1))
        return hidden_states
    
    def compute_rest_stage(self,hidden_states):
        for layer in self.model.wav2vec2.encoder.layers:
            layer_outputs = layer(
                        hidden_states, attention_mask=None, output_attentions=False
                    )
            hidden_states = layer_outputs[0]
        hidden_states = self.model.wav2vec2.encoder.layer_norm(hidden_states)
        # output = self.post_extractor(hidden_states)
        return hidden_states.mean(1)
    
    def forward(self,x):
        x = self.compute_stage1(x)
        x= self.compute_rest_stage(x)
        # output = self.post_extractor(x)
        return x
    