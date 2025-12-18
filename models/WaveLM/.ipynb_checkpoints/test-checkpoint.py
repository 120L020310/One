import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, WavLMModel

model = WavLMModel.from_pretrained(
    "/usr/local/ay_data/0-model_weights/microsoft_wavlm-base"
)

x = torch.randn(3, 48000)
model(x)['last_hidden_state'].shape

model.config

48000 / 5  / 2 / 2 / 2/2/2/2

extract_features = model.feature_extractor(x)
extract_features = extract_features.transpose(1, 2)
print(extract_features.shape)

# +
hidden_states, extract_features = model.feature_projection(extract_features)
hidden_states = model._mask_hidden_states(
    hidden_states, mask_time_indices=None, attention_mask=None
)

encoder_outputs = model.encoder(
    hidden_states,
    attention_mask=None,
    output_attentions=False,
    output_hidden_states=False,
    return_dict=False,
)
print(encoder_outputs[0].shape)
# -

model = model.cuda()
x = x.cuda()
model(x)


class BaseLine(nn.Module):
    def __init__(self, pretrain_feat="extract_features", backend='linear', num_classes=1, **kwargs):
        super().__init__()

        assert pretrain_feat in ["last_hidden_state", "extract_features"]
        self.pretrain_feat = pretrain_feat
        # The channels of used features for the pretrained model is 512 when using
        # the 'extract_features',  but 768 when ["last_hidden_state"] is used.
        C_features = 512 if pretrain_feat == "extract_features" else 768
        
        self.pretrain_model = WavLMModel.from_pretrained(
            "/usr/local/ay_data/0-model_weights/microsoft_wavlm-base"
        )

        self.backend = backend
        if backend == 'resnet':
            self.backend_model = ResNet50(
                in_channels=C_features, classes=num_classes
            )
        elif backend == 'linear':
            self.pooler = nn.AdaptiveAvgPool1d(1)
            self.backend_model = nn.Linear(C_features, num_classes)

    def get_feature(self, x):
        feature = self.pretrain_model(x)[self.pretrain_feat]
        feature = torch.transpose(feature, 1, 2)
        if self.backend == 'linear':
            feature = torch.squeeze(self.pooler(feature), -1)
        return feature
    
    def forward(self, x):
        feature = self.pretrain_model(x)[self.pretrain_feat]
        feature = torch.transpose(feature, 1, 2)
        if self.backend == 'linear':
            feature = torch.squeeze(self.pooler(feature), -1)
        # print(feature.shape, self.pooler(feature).shape)
        outputs = self.backend_model(feature)
        return outputs

    def extract_feature(self, x):
        return self.get_feature(x)

    def make_prediction(self, feature):
        outputs = self.backend_model(feature)
        return outputs
