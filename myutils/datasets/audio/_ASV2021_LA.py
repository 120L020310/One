# +
import os
import random
import re
from argparse import Namespace
from enum import Enum
from typing import NamedTuple, Union
from IPython.display import HTML, display

import numpy as np
import pandas as pd
from pandarallel import pandarallel
from tqdm.auto import tqdm

# -

from myutils.datasets.base import AudioDataset
from myutils.tools import read_file_paths_from_folder, to_list

VOCODERs = ["bonafide"] + ["A{:02d}".format(i) for i in range(7, 20)]


class ASV2021LA_AudioDs(AudioDataset):

    def update_audio_path(self, data, root_path):
        data["audio_path"] = data["relative_path"].apply(
            lambda x: os.path.join(root_path, x)
        )

    def postprocess(self):
        self.update_audio_path(self.data, self.root_path)
        self.vocoders = VOCODERs

    def read_label_data(self, root_path):
        label_files = ["keys/LA/CM/trial_metadata.txt"]

        label_data = pd.concat(
            [
                pd.read_csv(
                    os.path.join(root_path, _f),
                    delimiter=" ",
                    names=[
                        "speaker",
                        "filename",
                        "codec",
                        "col1",
                        "method",
                        "label",
                        "trim",
                        "split",
                    ],
                )
                for _f in label_files
            ],
            ignore_index=True,
        )
        ### convert label 'bonafide' -> 1, 'spoof' -> 0
        label_data["label"] = (
            label_data["label"].replace("bonafide", 1).replace("spoof", 0)
        )
        label_data["split"] = (
            label_data["split"].replace("progress", "train").replace("eval", "test")
        )

        ### randomly select 2000 samples from training split for validation.
        val_data = label_data.query("split == 'train'").sample(2000, random_state=42)
        label_data.loc[val_data.index, "split"] = "val"

        return label_data

    def _read_metadata(self, root_path, *args, **kwargs):
        paths = read_file_paths_from_folder(root_path, exts="flac")
        data = pd.DataFrame(paths, columns=["path"])
        data["relative_path"] = data["path"].apply(
            lambda x: x.replace(root_path + "/", "")
        )
        data["filename"] = data["path"].apply(
            lambda x: os.path.split(x)[1].replace(".flac", "")
        )

        label_data = self.read_label_data(root_path)
        data = pd.merge(data, label_data)

        data["vocoder_label"] = data["method"].apply(lambda x: VOCODERs.index(x))

        self.update_audio_path(data, root_path)
        data = self.read_audio_info(data)  # read fps and length
        return data

    def get_splits(self):
        """
            Get train/val/test splits according to the language.

        Args:

        Returns:
            Namespace(train, val, test)
        """

        data = self.data

        sub_datas = []
        for split in ["train", "val", "test"]:
            _data = data.query(f'split == "{split}"').reset_index(drop=True)
            sub_datas.append(_data)

        return Namespace(
            train=sub_datas[0],
            val=sub_datas[1],
            test=sub_datas[2],
        )
    
    def get_fake_train(self, data=None):
        data = data if data is not None else self.data.query("split == 'train'")
        sub_datas = []
        for vocoder in VOCODERs[1:]:
            _data = data.query(
                f"method == '{vocoder}'"
            ).reset_index(drop=True)
            sub_datas.append(_data)
        sub_datas = pd.concat(sub_datas, axis=0, ignore_index=True)
        return sub_datas
    
    def get_splits_val(self):
        data = self.data
        sub_datas = []
        for split in ["train", "val", "test"]:
            _data = data.query(f'split == "{split}"').reset_index(drop=True)
            sub_datas.append(_data)
        return sub_datas[1]

    def get_true_train(self, data=None):
        data = data if data is not None else self.data.query("split == 'train'")
        sub_datas = []
        for vocoder in VOCODERs[:1]:
            _data = data.query(
                f"method == '{vocoder}'"
            ).reset_index(drop=True)
            sub_datas.append(_data)
        sub_datas = pd.concat(sub_datas, axis=0, ignore_index=True)
        return sub_datas
    
    def get_test_splits_for_vocoder_types_T_SNE(self, data=None, add_bonafide=False):
        """split test data into 3 parts: vc, tts, at

        Args:
            add_bonafide (bool, optional): Defaults to True.

        Returns:
            list: [vc, tts, at]
        """
        data = self.data if data is None else data
        # if add_bonafide:
        #     vc_list = [0, 13, 15, 16, 24, 25, 26]
        #     tts_list = [0, 9, 10, 11, 12, 14, 17, 19, 21, 22, 28, 29]
        #     at_list = [0, 18, 20, 23, 27, 30, 31, 32]
        # else:
        #     vc_list = [13, 15, 16, 24, 25, 26]
        #     tts_list = [9, 10, 11, 12, 14, 17, 19, 21, 22, 28, 29]
        #     at_list = [18, 20, 23, 27, 30, 31, 32]
        #     bona_list = [0]

        # 创建分类掩码
        # vc_mask = data['vocoder_label'].isin(vc_list)
        # tts_mask = data['vocoder_label'].isin(tts_list)
        # at_mask = data['vocoder_label'].isin(at_list)
        bona_mask = data['label'].isin([1])
        fake_mask = data['label'].isin([0])
        
        # 划分数据集
        bona_data = data[bona_mask]
        fake_data = data[fake_mask]
        bona_data['vocoder_label'] = 0
        fake_data['vocoder_label'] = 1
        # vc_data = data[vc_mask]
        # vc_data['vocoder_label'] = 1
        # tts_data = data[tts_mask]
        # tts_data['vocoder_label'] = 2
        # at_data = data[at_mask]
        # at_data['vocoder_label'] = 3

        # # 验证划分结果
        # print(f"VC 样本数: {len(vc_data)}")
        # print(f"TTS 样本数: {len(tts_data)}")
        # print(f"AT 样本数: {len(at_data)}")
        sub_data=[]
        sub_data.append(bona_data)
        sub_data.append(fake_data)
        # sub_data.append(tts_data)
        # sub_data.append(at_data)
        return sub_data


# +
# root_path = "/home/ay/data/0-原始数据集/ASV2021-LA"
# ds = ASV2021LA_AudioDs(root_path=root_path)
# data = ds.data
# splits = ds.get_splits()
# +
# data.groupby(['split', 'label']).count()

# data.groupby(['split', 'method']).count()
# -
