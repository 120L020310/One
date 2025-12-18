import yaml
import os

from myutils.torch.deepfake_detection.audio import DeepfakeAudioClassification
from controlled_ex.RawGat_ST.model import RawGAT_ST
import torch
from copy import deepcopy
import torch.nn as nn


try:
    cwd = os.path.dirname(os.path.abspath(__file__))
except NameError:
    from pathlib import Path
    cwd = str(Path.cwd())


dir_yaml = os.path.join(cwd, "model_config_RawGAT_ST.yaml")
with open(dir_yaml, 'r') as f_yaml:
    parser1 = yaml.load(f_yaml, Loader=yaml.FullLoader)



class RawGAT_lit(DeepfakeAudioClassification):
    def __init__(self, **kwargs):
        super().__init__()
        model_config = deepcopy(parser1['model'])


        ## the model final output is [B, 1], not [B, 2]. thus we eliminate the pos weight in BCE loss.
        self.model = RawGAT_ST(model_config, 'cpu', num_classes=1)
        
        # self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([0.1, 0.9]))
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=None)
        self.save_hyperparameters()
    
    def calcuate_loss(self, batch_res, batch):
        label = batch["label"]
        loss = self.loss_fn(batch_res["logit"], label.type(torch.float32))
        return loss

    def _shared_pred(self, batch, batch_idx):
        audio, sample_rate = batch["audio"], batch["sample_rate"]
        if len(audio.shape) == 3:
            audio = audio[:, 0, :]

        out, feature = self.model(audio)
        out = out.squeeze(-1)
        batch_pred = (torch.sigmoid(out) + 0.5).int()
        return {
            "logit": out,
            "pred": batch_pred,
            "feature":feature
        }

    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=0.0001, weight_decay=0.0001
        )
        return [optimizer]
    
    
