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

from myutils.datasets.base import AudioDataset
VOCODERs = [
    "bonafide"
] + ["A%02d"% i for i in range(1, 33)]

COMPRESSIONs = [
    "-",
    "C01",
    "C02",
    "C03",
    "C04",
    "C05",
    "C06",
    "C07",
    "C08",
    "C09",
    "C10",
    "C11",
]

def init_track1_data(root_path):
    data = pd.read_csv(os.path.join(root_path, "metadata/track1.csv"), sep=",",low_memory=False)
    data["file"] = data["FLAC_FILE_NAME"].apply(
        lambda x: (
            f"flac_D/{x}.flac"
            if x.startswith("D")
            else (f"flac_E_eval/{x}.flac" if x.startswith("E") else f"flac_T/{x}.flac")
        )
    )
    data["vocoder"] = data["ATTACK_LABEL"]
    data["vocoder_label"] = data["vocoder"].apply(lambda x: VOCODERs.index(x))
    data["audio_path"] = data["file"].apply(lambda x: os.path.join(root_path, x))
    data["compression"] = data["CODEC"]
    data["compression_label"] = data["compression"].apply(lambda x: COMPRESSIONs.index(x))
    data["compression_quality"] = data["CODEC_Q"].replace("-", 0)
    return data

class ASVSpoof5_AudioDs(AudioDataset):
    def postprocess(self):
        self.data["audio_path"] = self.data["file"].apply(
            lambda x: os.path.join(self.root_path, x)
        )
        
        self.vocoders = VOCODERs
        self.compressions = COMPRESSIONs


    def read_metadata(self, root_path, *args, **kwargs):
        data = init_track1_data(root_path)
        return data


    def split_train_val_in_train_tsv(self, train_data=None, train_val_rate_in_train_tsv=0.8):
        from myutils.tools.pandas import DF_spliter

        if train_data is None:
            data = self.data
            train_data = data[data['split'] == 'train']

        
        train, val = DF_spliter.split_by_number_and_column(train_data, [0.8, 0.2], refer='SPEAKER_ID')

        ids_train = set(list(train['SPEAKER_ID']))
        ids_val = set(list(val['SPEAKER_ID']))
        print(
            "###########"*9 + '\n',
            "Split train_data of ASVspoof5 track1 into train/val subsets, total 400 speakers.\n"
            f"After spliting, there are {len(ids_train)} speakers and {len(train)} audios in train subset\n"
            f"                there are {len(ids_val)} speakers and {len(val)} audios in val subset\n"
            f"Is speakers in train and val subset disjoint? ->{ids_train.isdisjoint(ids_val)} \n",
            "###########"*9
            )
        return train, val


    # def get_splits(self, train_val_rate_in_train_tsv=0.8, use_dev_as_test=True, use_both_dev_test_for_test=False):
    #     """
    #         Get train/val/test splits according to the language.
    #     Args:

    #     Returns:
    #         Namespace(train, val, test)
    #     """

    #     data = self.data


    #     train = data.query("split == 'train'").reset_index(drop=True)
    #     train, val = self.split_train_val_in_train_tsv(
    #         train_data=train,
    #         train_val_rate_in_train_tsv=train_val_rate_in_train_tsv
    #     )
    #     test = data.query("split == 'test'").reset_index(drop=True)
    #     dev = data.query("split == 'dev'").reset_index(drop=True)
        
        
        
    #     res = Namespace(train=train, val=val)
        
    #     if use_dev_as_test:
    #         res.test = dev
    #     elif use_both_dev_test_for_test:
    #         res.test=[dev, test, pd.concat([dev, test], ignore_index=True)]
    #     else:
    #         res.test = test
    #     return res
    
    def get_test_splits(self, data=None):
        """
            Get train/val/test splits according to the language.

        Args:

        Returns:
            Namespace(train, val, test)
        """

        data = data if data is not None else self.data.query("split == 'test'")

        sub_datas = []
        for vocoder in VOCODERs[1:]:
            _data = data.query(
                f"vocoder == '{vocoder}' or vocoder == '{VOCODERs[0]}'"
            ).reset_index(drop=True)
            sub_datas.append(_data)

        return sub_datas
    
    # def get_vocoder_test_splits(self, add_bonafide=True):
    #     data = self.data
    #     if add_bonafide:
    #         vc_list = [0, 13, 15, 16, 24, 25, 26]
    #         tts_list = [0, 9, 10, 11, 12, 14, 17, 19, 21, 22, 28, 29]
    #         at_list = [0, 18, 20, 23, 27, 30, 31, 32]
    #     else:
    #         vc_list = [13, 15, 16, 24, 25, 26]
    #         tts_list = [9, 10, 11, 12, 14, 17, 19, 21, 22, 28, 29]
    #         at_list = [18, 20, 23, 27, 30, 31, 32]

    #     # 创建分类掩码
    #     vc_mask = data['vocoder_label'].isin(vc_list)
    #     tts_mask = data['vocoder_label'].isin(tts_list)
    #     at_mask = data['vocoder_label'].isin(at_list)
        
    #     # 划分数据集
    #     vc_data = data[vc_mask]
    #     tts_data = data[tts_mask]
    #     at_data = data[at_mask]

    #     # # 验证划分结果
    #     # print(f"VC 样本数: {len(vc_data)}")
    #     # print(f"TTS 样本数: {len(tts_data)}")
    #     # print(f"AT 样本数: {len(at_data)}")
    #     sub_data=[]
    #     sub_data.append(vc_data)
    #     sub_data.append(tts_data)
    #     sub_data.append(at_data)
    #     return sub_data
    def get_true_train(self, data=None):
        data = data if data is not None else self.data.query("split == 'train'")
        sub_datas = []
        for vocoder in VOCODERs[:1]:
            _data = data.query(
                f"label == 1"
            ).reset_index(drop=True)
            sub_datas.append(_data)
        sub_datas = pd.concat(sub_datas, axis=0, ignore_index=True)
        return sub_datas

    def get_splits(self, train_val_rate_in_train_tsv=0.8, 
                   use_dev_as_test=False, 
                   use_both_dev_test_for_test=False,
                   only_test_vocoder=False):
        """
            Get train/val/test splits according to the language.
        Args:

        Returns:
            Namespace(train, val, test)
        """

        data = self.data


        train = data.query("split == 'train'").reset_index(drop=True)
        train, val = self.split_train_val_in_train_tsv(
            train_data=train,
            train_val_rate_in_train_tsv=train_val_rate_in_train_tsv
        )
        test = data.query("split == 'test'").reset_index(drop=True)
        dev = data.query("split == 'dev'").reset_index(drop=True)
        
        
        
        res = Namespace(train=train, val=val)
        
        if use_dev_as_test:
            res.test = dev
        elif use_both_dev_test_for_test:
            all_test_data = pd.concat([dev, test], ignore_index=True)
            res.test=[dev, test, all_test_data] + self.get_test_splits_for_vocoder_types(all_test_data)
        elif only_test_vocoder:
            all_test_data = pd.concat([dev, test], ignore_index=True)
            res.test= self.get_test_splits_for_vocoder_types(all_test_data)
        else:
            res.test = test
        return res
    
    
    def get_test_splits_for_vocoders(self, data=None):
        """
            Get train/val/test splits according to the vocoder.

        Args:

        Returns:
            Namespace(train, val, test)
        """

        data = data if data is not None else self.data.query("split == 'test'")

        sub_datas = []
        for vocoder in VOCODERs[1:]:
            _data = data.query(
                f"vocoder == '{vocoder}' or vocoder == '{VOCODERs[0]}'"
            ).reset_index(drop=True)
            sub_datas.append(_data)

        return sub_datas
    
    
    def get_test_splits_for_vocoder_types(self, data=None, add_bonafide=True):
        """split test data into 3 parts: vc, tts, at

        Args:
            add_bonafide (bool, optional): Defaults to True.

        Returns:
            list: [vc, tts, at]
        """
        data = self.data if data is None else data
        if add_bonafide:
            vc_list = [0, 13, 15, 16, 24, 25, 26]
            tts_list = [0, 9, 10, 11, 12, 14, 17, 19, 21, 22, 28, 29]
            at_list = [0, 18, 20, 23, 27, 30, 31, 32]
        else:
            vc_list = [13, 15, 16, 24, 25, 26]
            tts_list = [9, 10, 11, 12, 14, 17, 19, 21, 22, 28, 29]
            at_list = [18, 20, 23, 27, 30, 31, 32]

        # 创建分类掩码
        vc_mask = data['vocoder_label'].isin(vc_list)
        tts_mask = data['vocoder_label'].isin(tts_list)
        at_mask = data['vocoder_label'].isin(at_list)
        
        # 划分数据集
        vc_data = data[vc_mask]
        tts_data = data[tts_mask]
        at_data = data[at_mask]

        # # 验证划分结果
        # print(f"VC 样本数: {len(vc_data)}")
        # print(f"TTS 样本数: {len(tts_data)}")
        # print(f"AT 样本数: {len(at_data)}")
        sub_data=[]
        sub_data.append(vc_data)
        sub_data.append(tts_data)
        sub_data.append(at_data)
        return sub_data
    
    def get_test_splits_for_vocoder_types_T_SNE(self, data=None, add_bonafide=False):
        """split test data into 3 parts: vc, tts, at

        Args:
            add_bonafide (bool, optional): Defaults to True.

        Returns:
            list: [vc, tts, at]
        """
        data = self.data if data is None else data
        if add_bonafide:
            vc_list = [0, 13, 15, 16, 24, 25, 26]
            tts_list = [0, 9, 10, 11, 12, 14, 17, 19, 21, 22, 28, 29]
            at_list = [0, 18, 20, 23, 27, 30, 31, 32]
        else:
            vc_list = [13, 15, 16, 24, 25, 26]
            tts_list = [9, 10, 11, 12, 14, 17, 19, 21, 22, 28, 29]
            at_list = [18, 20, 23, 27, 30, 31, 32]
            bona_list = [0]

        # 创建分类掩码
        vc_mask = data['vocoder_label'].isin(vc_list)
        tts_mask = data['vocoder_label'].isin(tts_list)
        at_mask = data['vocoder_label'].isin(at_list)
        bona_mask = data['vocoder_label'].isin(bona_list)
        
        # 划分数据集
        bona_data = data[bona_mask]
        bona_data['vocoder_label'] = 0
        vc_data = data[vc_mask]
        vc_data['vocoder_label'] = 1
        tts_data = data[tts_mask]
        tts_data['vocoder_label'] = 2
        at_data = data[at_mask]
        at_data['vocoder_label'] = 3

        # # 验证划分结果
        # print(f"VC 样本数: {len(vc_data)}")
        # print(f"TTS 样本数: {len(tts_data)}")
        # print(f"AT 样本数: {len(at_data)}")
        sub_data=[]
        sub_data.append(bona_data)
        sub_data.append(vc_data)
        sub_data.append(tts_data)
        sub_data.append(at_data)
        return sub_data