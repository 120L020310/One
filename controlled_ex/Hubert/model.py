import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from myutils.tools import freeze_modules
from myutils.torch.nn import LambdaFunctionModule
from einops import rearrange
from transformers import AutoProcessor, HubertModel
# -

# try:
from models.WaveLM.wavlm import BaseLine as WavLM
from myutils.zyz.my_utils import post_process

# except ImportError:
#     sys.path.append("../../WaveLM")
#     from wavlm import BaseLine as WavLM

class Hubert_ASR(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.model = HubertModel.from_pretrained("/home/zyz/data/hubert_pretrained")
        # self.post_extractor = post_process(hidden_dim=16709)
    def compute_stage1(self,x):
        x = x.squeeze(1)
        extract_features = self.model.feature_extractor(x)
        extract_features = extract_features.transpose(1, 2)
        hidden_states = self.model.feature_projection(extract_features)
        position_embeddings = self.model.encoder.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.model.encoder.dropout(hidden_states)
        return hidden_states
    
    def compute_rest_stage(self,hidden_states):
        for layer in self.model.encoder.layers:
            layer_outputs = layer(
                        hidden_states, attention_mask=None, output_attentions=False
                    )
            hidden_states = layer_outputs[0]
        hidden_states = self.model.encoder.layer_norm(hidden_states)
        output = self.post_extractor(hidden_states)
        return output
    
    def forward(self,x):
        x = x.squeeze(1)
        extract_features = self.model.feature_extractor(x)
        extract_features = extract_features.transpose(1, 2)
        hidden_states = self.model.feature_projection(extract_features)
        position_embeddings = self.model.encoder.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.model.encoder.dropout(hidden_states)
        for layer in self.model.encoder.layers:
            layer_outputs = layer(
                        hidden_states, attention_mask=None, output_attentions=False
                    )
            hidden_states = layer_outputs[0]
        hidden_states = self.model.encoder.layer_norm(hidden_states)
        return hidden_states.mean(1)