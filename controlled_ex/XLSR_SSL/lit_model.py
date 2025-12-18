
import pytorch_lightning as pl
import torch
import torch.nn as nn

from myutils.torch.deepfake_detection.audio import DeepfakeAudioClassification

from .model import XLS_R,SSLModel


class XLS_R_lit(DeepfakeAudioClassification):
    
    
    def __init__(self, cfg=None, args=None, **kwargs):
        super().__init__()
        
        self.cfg = cfg
        self.args = args
        self.model = SSLModel()
        self.configure_loss_fn()
        self.configure_normalizer()
        
    
    def configure_loss_fn(self):
        self.loss_fn = nn.BCEWithLogitsLoss()
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-6, weight_decay=1e-4)
        return optimizer

    def calcuate_loss(self, batch_res, batch):
        label = batch["label"]
        loss = self.loss_fn(batch_res["logit"], label.type(torch.float32))
        return loss


    def _shared_pred(self, batch, batch_idx, stage='train', **kwargs):
        audio = batch["audio"]
        if len(audio.shape) == 3:
            audio = audio[:, 0, :]

        res = self.model(audio, return_dict=True)
        
        out = res["logit"]
        latent_feat = res["latent_feat"]        
        
        # if out.shape[1] == 1:
        #     out = out.squeeze(-1)

        
        # now out is a 1D tensor with shape (batch_size,)
        batch_pred = (torch.sigmoid(out) + 0.5).int()
        
        return {
            "logit": out,
            "org_logit": res["logit"],
            "pred": batch_pred,
            "feature": latent_feat
        }
