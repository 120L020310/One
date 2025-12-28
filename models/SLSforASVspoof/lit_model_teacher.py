
import pytorch_lightning as pl
import torch
import torch.nn as nn

from models.SLSforASVspoof.teacher import XLSR_Teacher
from models.SLSforASVspoof.teacher_strict import ALDA_Teacher_Strict
from myutils.torch.deepfake_detection.audio import DeepfakeAudioClassification
from myutils.zyz.aocloss import AOCloss

from .model import XLS_R_SLS


class ALDA_teacher_lit(DeepfakeAudioClassification):
    
    
    def __init__(self, cfg=None, args=None, **kwargs):
        super().__init__()
        
        self.cfg = cfg
        self.args = args
        self.model = XLSR_Teacher()
        self.configure_loss_fn()
        self.configure_normalizer()
        self.cls_h = torch.nn.Linear(1024,2)
        
    
    def configure_loss_fn(self):
        self.loss_fn = nn.CrossEntropyLoss()
        
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

        batch_res = self.model(audio)
        out = self.cls_h(batch_res["final_feat"].mean(1))
        # out = res["logit"]
        # latent_feat = res["latent_feat"]        
        
        if out.shape[1] == 2:
            out = out[:, 1] - out[:, 0]
        elif out.shape[1] == 1:
            out = out.squeeze(-1)
        else:
            raise ValueError(f"Invalid output shape: {out.shape}, expected 1 or 2 classes.")
        
        # now out is a 1D tensor with shape (batch_size,)
        batch_pred = (torch.sigmoid(out) + 0.5).int()
        
        return {
            "logit": out,
            "pred": batch_pred,
        }