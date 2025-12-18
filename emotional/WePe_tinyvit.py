import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2Processor
from transformers.models.wav2vec2.modeling_wav2vec2 import (
    Wav2Vec2Model,
    Wav2Vec2PreTrainedModel,
)

from myutils.torch.deepfake_detection.audio import DeepfakeAudioClassification



class Adapter(nn.Module):
    """
    Bottleneck adapter: down -> nonlinear -> up
    初始化为近似 identity（DFM要求：训练初期不破坏原特征）
    """
    def __init__(self, hidden_dim=1024, bottleneck_dim=256):
        super().__init__()
        self.down = nn.Linear(hidden_dim, bottleneck_dim)
        self.act = nn.ReLU()
        self.up = nn.Linear(bottleneck_dim, hidden_dim)

        # 初始化为近似 identity
        nn.init.zeros_(self.down.weight)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, x):
        """
        x: (batch, seq_len, hidden_dim)
        """
        z = self.down(x)
        z = self.act(z)
        z = self.up(z)
        return x + z   # 保持 residual，稳定训练
    
class EmotionModel_adapter(DeepfakeAudioClassification):
    """
    Wav2Vec2 + External Adapters for DFM-style training.
    不修改 Wav2Vec2Model 内部层，完全外部注入。
    """

    def __init__(self, adapter_dim=256):
        super().__init__()
        model_name = '/home/zyz/data/work2/wav2vec2-large-robust-12-ft-emotion-msp-dim'
        self.model = EmotionModel.from_pretrained(model_name)

        # backbone hidden dimension
        hidden_dim = self.model.wav2vec2.config.hidden_size  # typically 1024

        # 为每个 transformer encoder 层注入 adapter
        num_layers = len(self.model.wav2vec2.encoder.layers)
        self.adapters = nn.ModuleList([
            Adapter(hidden_dim, adapter_dim) for _ in range(num_layers)
        ])

        self.model.init_weights()

    def forward(self, input_values,adapter=False):
        """
        输入：raw waveform  (batch, 48000)
        输出：embedding     (batch, hidden_dim)
        """
        extract_features = self.model.wav2vec2.feature_extractor(input_values)
        extract_features = extract_features.transpose(1, 2)
        hidden_states, extract_features = self.model.wav2vec2.feature_projection(extract_features)
        position_embeddings = self.model.wav2vec2.encoder.pos_conv_embed(hidden_states)
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.model.wav2vec2.encoder.dropout(hidden_states)

        # --------------------
        # 3. Transformer encoder with external adapters
        # --------------------
        for layer_idx, layer in enumerate(self.model.wav2vec2.encoder.layers):
            if adapter:
                hidden_states = self.adapters[layer_idx](hidden_states)
            hidden_states = layer(hidden_states)[0]
        hidden_states = self.model.wav2vec2.encoder.layer_norm(hidden_states)
        pooled = hidden_states.mean(dim=1)

        return pooled


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

        outputs = self.wav2vec2(input_values)
        hidden_states = outputs[0]
        hidden_states = torch.mean(hidden_states, dim=1)
        # logits = self.classifier(hidden_states)

        return hidden_states






class test_lit(DeepfakeAudioClassification):
    def __init__(self, **kwargs):
        super().__init__()

        model_name = '/home/zyz/data/work2/wav2vec2-large-robust-12-ft-emotion-msp-dim'
        self.model = EmotionModel.from_pretrained(model_name)
        print(self.model)

    def forward(self, x):
        y = self.model(x.squeeze(1))
        return {"final_feat":y}