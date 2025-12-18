# +
import sys

sys.path.append("../../")
from data.tools import WaveDataset

# +
from argparse import Namespace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from ay2.datasets import ASVP_ESD, VGG_Sound
from ay2.tools.pandas import DF_spliter
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import default_collate
# -


def collate_fn(batch):
    res = {}

    audio_list = [x["audio"].squeeze() for x in batch]
    padded_audio = pad_sequence(audio_list, batch_first=True, padding_value=0)

    for x in batch:
        x.pop("audio")

    default_res = default_collate(batch)
    default_res["audio"] = padded_audio
    return default_res


def get_data():
    dataset = VGG_Sound()
    data = dataset.data

    datasets = []
    data_loaders = []
    for i, _d in enumerate(DF_spliter.split_by_percentage(data, splits=[0.9, 0.1])):
        print(i, len(_d), len(data))
        _ds = WaveDataset(
            _d,
            sample_rate=16000,
            trim=False,
            max_wave_length=48000,
            transform=None,
            is_training=True if i == 0 else False,
        )
        _dl = torch.utils.data.DataLoader(
            _ds,
            batch_size=64,
            num_workers=4,
            pin_memory=True,
            shuffle=True,
            prefetch_factor=2,
            # collate_fn=collate_fn,
        )
        data_loaders.append(_dl)

    dl = Namespace(train=data_loaders[0], val=data_loaders[1], test=data_loaders[-1])
    return dl
