# %load_ext autoreload
# %autoreload 2

import pytorch_lightning as pl
import torch
import numpy as np
import math
import torch.nn as nn
from myutils.torch.deepfake_detection import DeepfakeAudioClassification

from einops import rearrange
from torchaudio.transforms import LFCC

try:
    from grl import GradientReversal
    from lcnn import LCNN
except ImportError:
    from .grl import GradientReversal
    from .lcnn import LCNN


class BLSTM(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.drop = nn.Dropout2d(0.1)
        self.rnn = nn.LSTM(input_size, input_size // 2, num_layers=2, bidirectional=True, batch_first=True)

    def forward(self, x):
        x = self.drop(x)
        x = rearrange(x, "b c h w -> b h (c w)")
        
        output, _ = self.rnn(x)
        return output + x


class ASDG(nn.Module):
    def __init__(self, n_LFCC=80):
        super().__init__()

        n_dim = n_LFCC // 16 * 32
        self.lfcc = LFCC(
            n_lfcc=n_LFCC,
            speckwargs={"n_fft": 400, "hop_length": 160, "center": True},
        )
        self.lcnn = LCNN()
        self.lstm = BLSTM(n_dim)
        self.fc1 = nn.Linear(n_dim, 512)
        self.fc2 = nn.utils.parametrizations.weight_norm(nn.Linear(512, 1))

        self.grl = GradientReversal()
        self.fc_domain = nn.utils.parametrizations.weight_norm(nn.Linear(512, 3))

    def extract_feature(self, x, debug=0):
        lfcc = self.lfcc(x)
        lfcc = rearrange(lfcc, "b c h w -> b c w h")

        x = self.lcnn.extract_feature(lfcc, debug=debug)  # (B, 32, W, n_LFCC//16)

        x = self.lstm(x)  # (B, W, 32 * n_LFCC//16)
        x = torch.mean(x, dim=1)  # (B, 32 * n_LFCC//16)
        feat = self.fc1(x)  # (B, 512)
        feat = feat / (1e-9 + torch.norm(feat, p=2, dim=-1, keepdim=True))

        return feat

    def make_prediction(self, feat):
        logit = self.fc2(feat)
        return logit

    def gene_grl_alpha(self, current_step, total_step):
        a = 1 - 2 / (1 + math.exp(-10 * current_step / total_step))
        return torch.tensor(a)
    
    def predict_domain_label(self, feat, current_step=0, total_step=1):
        alpha = self.gene_grl_alpha(current_step, total_step)
        feat_rev = self.grl(feat, alpha=alpha)
        logit = self.fc_domain(feat_rev)
        return logit

    def forward(self, x, debug=0):
        feat = self.extract_feature(x, debug=debug)
        logit = self.fc2(feat)
        return logit


# +
# model = ASDG()
# x = torch.randn(2, 1, 16000 * 3)
# model(x)
# -

class ASDG_lit(DeepfakeAudioClassification):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = ASDG(n_LFCC=80)

        self.configure_loss_fn()

        self.save_hyperparameters()

    def configure_loss_fn(self):
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.triplet_loss = nn.TripletMarginLoss(margin=0.1, p=2, eps=1e-7)

    
    def get_domain_label(self, batch):

        domains = ['en', 'de', 'es']
        batch['ASGD_domain_label'] = []
        for i in range(len(batch['label'])):
            try:
                _label = batch['language'][i]
            except KeyError:
                _label = domains[0]
            if _label in domains:
                _label = domains.index(_label)
            else:
                _label = 0
            batch['ASGD_domain_label'].append(_label)
        batch['ASGD_domain_label'] = torch.tensor(batch['ASGD_domain_label']).to(batch['label'].device)
    
    def generate_pairs(self, features, labels):
        """
        Generate positive and negative pairs based on class labels.
        
        Args:
        - features (torch.Tensor): A tensor of shape (batch_size, feature_dim) representing the batch of features.
        - labels (torch.Tensor): A tensor of shape (batch_size,) representing the class labels for each feature.
        
        Returns:
        - pos_pairs (list of tuples): List of positive pairs (anchor_idx, pos_idx).
        - neg_pairs (list of tuples): List of negative pairs (anchor_idx, neg_idx).
        """
        pos_pairs = []
        neg_pairs = []
        batch_size = features.size(0)
        
        for anchor_idx in range(batch_size):
            anchor_label = labels[anchor_idx]

            find_pos = 0
            find_neg = 0
            
            for pair_idx in np.random.permutation(np.arange(batch_size)):
                if anchor_idx == pair_idx:
                    continue  # Skip the same element
                
                pair_label = labels[pair_idx]
                
                if anchor_label == pair_label:
                    if not find_pos:
                        pos_pairs.append(features[pair_idx])
                        find_pos = 1
                else:
                    if not find_neg:
                        neg_pairs.append(features[pair_idx])
                        find_neg = 1
                if find_pos and find_neg:
                    break
            if not find_pos:
                pos_pairs.append(features[anchor_idx])
            if not find_neg:
                neg_pairs.append(features[anchor_idx])

        pos_pairs = torch.stack(pos_pairs)
        neg_pairs = torch.stack(neg_pairs)

        # print(pos_pairs.shape, neg_pairs.shape)
        
        return pos_pairs, neg_pairs
    
    
    def calcuate_loss(self, batch_res, batch):
        self.get_domain_label(batch)
        label = batch["label"]
        batch_size = len(label)
        loss1 = self.bce_loss(batch_res["logit"], label.type(torch.float32))

        idx = torch.nonzero(label==1).squeeze().cpu()
        if idx.ndim==0 or len(idx) == 0:
            adv_loss = 0
        else:
            adv_loss = self.ce_loss(batch_res['domain_logit'][idx], batch['ASGD_domain_label'][idx])

        anchor = batch_res['feature']
        triplet_label = []
        for i in range(batch_size):
            _label = 3 if label[i] == 1 else batch['ASGD_domain_label'][i]
            triplet_label.append(_label)
        pos_pairs, neg_pairs = self.generate_pairs(anchor, triplet_label)
        triplet_loss = self.triplet_loss(anchor, pos_pairs, neg_pairs)

        loss = loss1 + 0.1 * (adv_loss + triplet_loss)
        return {
            "loss" : loss,
            "cls_loss" : loss1,
            "adv_loss" : adv_loss,
            "triplet_loss" : triplet_loss
        }

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.0001, weight_decay=0.0001)
        self.num_training_batches = self.trainer.num_training_batches
        return [optimizer]

    def _shared_pred(self, batch, batch_idx):
        audio, sample_rate = batch["audio"], batch["sample_rate"]

        # batch_out = self.model(audio).squeeze()
        feature = self.model.extract_feature(audio)
        batch_out = self.model.make_prediction(feature).squeeze(-1)
        batch_pred = (torch.sigmoid(batch_out) + 0.5).int()


        current_step = self.global_step
        try:
            total_step = self.num_training_batches * 64 * 50
        except AttributeError:
            total_step = 1000
        domain_logit = self.model.predict_domain_label(feature, current_step, total_step)
        
        return {"logit": batch_out, "domain_logit": domain_logit, "pred": batch_pred, "feature": feature}
    
    def forward(self, audio):
        # batch_out = self.model(audio).squeeze()
        feature = self.model.extract_feature(audio)

        return feature

