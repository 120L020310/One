import torch
import torch.nn as nn
import torch.nn.functional as F
import os



from transformers import AutoModelForPreTraining


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
        
        return feat, hidden_states, logit.squeeze(-1)

class SSLModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.src_model = XLS_R()
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.selu = nn.SELU(inplace=True)
        self.fc0 = nn.Linear(1024, 1)
        self.sig = nn.Sigmoid()
        self.fc1 = nn.Linear(16709, 1024)
        self.fc3 = nn.Linear(1024,2)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x,return_dict=False):
        layerResult, x_ssl_feat= self.src_model.extract_feat(x.squeeze(-1)) #layerresult = [(x,z),24个] x(201,1,1024) z(1,201,201)
        y0, fullfeature = getAttenF(layerResult)
        y0 = self.fc0(y0)
        y0 = self.sig(y0)
        y0 = y0.view(y0.shape[0], y0.shape[1], y0.shape[2], -1)
        fullfeature = fullfeature * y0
        fullfeature = torch.sum(fullfeature, 1)
        fullfeature = fullfeature.unsqueeze(dim=1)
        x = self.first_bn(fullfeature)
        x = self.selu(x)
        x = F.max_pool2d(x, (3, 3))
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.selu(x)
        x = self.fc3(x)
        x = self.selu(x)
        output = self.logsoftmax(x)
        # 计算一位logit
        logit = output[:, 1] - output[:, 0] 
        if return_dict:
            return {'logit': logit, 'latent_feat': x_ssl_feat}
        return logit

def getAttenF(layerResult):
    poollayerResult = []
    fullf = []
    for layer in layerResult:

        layery = layer.transpose(1, 2) #(x,z)  x(201,b,1024) (b,201,1024) (b,1024,201)
        layery = F.adaptive_avg_pool1d(layery, 1) #(b,1024,1)
        layery = layery.transpose(1, 2) # (b,1,1024)
        poollayerResult.append(layery)

        x = layer
        x = x.view(x.size(0), -1,x.size(1), x.size(2))
        fullf.append(x)

    layery = torch.cat(poollayerResult, dim=1)
    fullfeature = torch.cat(fullf, dim=1)
    return layery, fullfeature
