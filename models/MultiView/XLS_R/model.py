# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import os



# %%
from transformers import AutoModelForPreTraining


# %%
class XLS_R(nn.Module):
    
    
    def __init__(self, pretrained_path="/home/zyz/data/wav2vec2-xls-r-300m", audio_length=48000, **kwargs):
        super().__init__()

    
        # model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([pretrained_path])
        # self.ssl_model = model[0]
        self.ssl_model = AutoModelForPreTraining.from_pretrained(pretrained_path)
        print(self.ssl_model)
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.fc0 = nn.Linear(1024, 1)
        self.sig = nn.Sigmoid()
        if audio_length == 48000:
            self.fc1 = nn.Linear(16709, 1024)
        elif audio_length == 64000:
            self.fc1 = nn.Linear(22847, 1024)
        else:
            raise ValueError("audio_length should be 48000 or 64000, but got {}".format(audio_length))
        
        
        self.fc3 = nn.Linear(1024,1)
        self.logsoftmax = nn.LogSoftmax(dim=1)


    def extract_feat(self, input_data: torch.Tensor):
        
        res = self.ssl_model(input_data, output_hidden_states=True)
        hidden_states = res.hidden_states
        final_hidden_state = hidden_states[-1]
        return hidden_states[1:], final_hidden_state

    def forward(self, x, return_dict=False):
        x = x.squeeze(1)
        hidden_states, final_hidden_state = self.extract_feat(x)
        
        feat = final_hidden_state.mean(dim=1) #(batch_size, 1024)
        logit = self.fc3(feat)
        
        # x = self.first_bn(fullfeature)
        # x = self.selu(x)
        # x = F.max_pool2d(x, (3, 3))
        # x = torch.flatten(x, 1)
        # x = self.fc1(x)
        # feat = self.selu(x)
        # logit = self.logsoftmax(self.selu(self.fc3(feat)))

        if return_dict:
            return {'logit': logit, 'latent_feat': feat}
        
        return feat, hidden_states,logit.squeeze(-1)


# %%

# %%
# model = XLS_R_SLS()

# %%
# import torch
# x = torch.randn(3, 16000*3)
# model(x) # (3, 2)

# %%
model = XLS_R()