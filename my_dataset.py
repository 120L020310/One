import argparse

from config.config import get_cfg_defaults
from data.make_dataset import build_dataloader, build_transforms, make_data
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler, SequentialSampler, Dataset
from tqdm import tqdm

from data.tools.dataset import WaveDataset
from myutils.datasets.audio._ASV2021 import ASV2021_AudioDs
from myutils.datasets.audio._ASV2021_LA import ASV2021LA_AudioDs
from myutils.datasets.audio._ASV5 import ASVSpoof5_AudioDs
from myutils.datasets.audio._MLAAD import MLAAD_AudioDs

def get_dataset_and_loader_ASV2021_inner(label):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default=f"MultiView/ASV2021_inner")
    parser.add_argument("--test_noise", type=int, default=0)
    parser.add_argument("--gpu", type=int, nargs="+", default=0)
    parser.add_argument("-v", "--version", type=int, default=None)
    args = parser.parse_args()
    cfg = get_cfg_defaults(
        "config/experiments/%s.yaml" % args.cfg, ablation=None
    )
    asv2021 = ASV2021_AudioDs(root_path="/home/zyz/data/2021DF/ASVspoof2021_DF_eval")
    if label=="fake":   ds = asv2021.get_fake_train()
    elif label=="real": ds = asv2021.get_true_train()
    elif label=="val": ds = asv2021.get_splits_val()
    elif label=="split": ds = asv2021.get_splits()
    transforms = build_transforms(cfg.DATASET.transforms, args=args)
    if label=="split":
        ds_train = WaveDataset(data=ds.train,normalize=True,transform=transforms["train"])
        dl_train = DataLoader(ds_train, batch_size=32, shuffle=False, num_workers=8)
        ds_val = WaveDataset(data=ds.val,normalize=True,transform=transforms["val"])
        dl_val = DataLoader(ds_val, batch_size=32, shuffle=False, num_workers=8)
        test_loaders=[]
        for idx in range(len(ds.test)):
            # 1. 确保 MyDataset 实例接收的是单个 DataFrame
            ds_test = WaveDataset(data=ds.test[idx],normalize=True,transform=transforms["val"])
            # 2. 为每个数据集创建独立的 DataLoader
            loader = DataLoader(
                ds_test, 
                batch_size=32, 
                shuffle=False,  # 测试时通常不shuffle
                num_workers=8
            )
            test_loaders.append(loader)

        return dl_train,dl_val,test_loaders
    ds = WaveDataset(data=ds,normalize=True,transform=transforms["train"])
    dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=8)
    return ds,dl


def get_dataset_and_loader_ASV2021_LA(label):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default=f"MultiView/ASV2021_LA")
    parser.add_argument("--test_noise", type=int, default=0)
    parser.add_argument("--gpu", type=int, nargs="+", default=0)
    parser.add_argument("-v", "--version", type=int, default=None)
    args = parser.parse_args()
    cfg = get_cfg_defaults(
        "config/experiments/%s.yaml" % args.cfg, ablation=None
    )
    asv2021 = ASV2021LA_AudioDs(root_path="/home/zyz/data/LA_2021/ASVspoof2021_LA_eval")
    if label=="fake":   ds = asv2021.get_fake_train()
    elif label=="real": ds = asv2021.get_true_train()
    elif label=="val": ds = asv2021.get_splits_val()
    elif label=="split": ds = asv2021.get_splits()
    transforms = build_transforms(cfg.DATASET.transforms, args=args)
    if label=="split":
        ds_train = WaveDataset(data=ds.train,normalize=True,transform=transforms["train"])
        dl_train = DataLoader(ds_train, batch_size=32, shuffle=False, num_workers=8)
        ds_val = WaveDataset(data=ds.val,normalize=True,transform=transforms["val"])
        dl_val = DataLoader(ds_val, batch_size=32, shuffle=False, num_workers=8)
        ds_test = WaveDataset(data=ds.test,normalize=True,transform=transforms["val"])
        dl_test = DataLoader(ds_test, batch_size=32, shuffle=False, num_workers=8)

        return dl_train,dl_val,[dl_test]
    else:
        ds = WaveDataset(data=ds,normalize=True,transform=transforms["val"])
        dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=8)
        return ds, dl

# parser = argparse.ArgumentParser()
# parser.add_argument("--cfg", type=str, default=f"MultiView/ASV2021_inner")
# parser.add_argument("--test_noise", type=int, default=0)
# args = parser.parse_args()
# cfg = get_cfg_defaults(
#     "config/experiments/%s.yaml" % args.cfg, ablation=None
# )
# asv2021 = ASV2021_AudioDs(root_path="/home/zyz/data/2021DF/ASVspoof2021_DF_eval")
# ds = asv2021.get_fake_train()
# transforms = build_transforms(cfg.DATASET.transforms, args=args)
# transform = transforms["train"]
# ds = WaveDataset(data=ds,normalize=True,transform=transform)
# dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=8)
# print(dl)

def get_dataset_and_loader_ASV5(label):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default=f"MultiView/ASVSpoof5")
    parser.add_argument("--test_noise", type=int, default=0)
    args = parser.parse_args()
    cfg = get_cfg_defaults(
        "config/experiments/%s.yaml" % args.cfg, ablation=None
    )
    dataset = ASVSpoof5_AudioDs(root_path=cfg.DATASET.dataset_cfg.root_path)
    if label=="fake":   ds = dataset.get_splits(only_test_vocoder=True)
    # elif label=="real": ds = asv2021.get_true_train()
    # elif label=="val": ds = asv2021.get_splits_val()
    elif label=="split": ds = dataset.get_splits(only_test_vocoder=True)
    transforms = build_transforms(cfg.DATASET.transforms, args=args)
    if label=="split":
        ds_train = WaveDataset(data=ds.train,normalize=True,transform=transforms["train"])
        dl_train = DataLoader(ds_train, batch_size=32, shuffle=False, num_workers=8)
        ds_val = WaveDataset(data=ds.val,normalize=True,transform=transforms["val"])
        dl_val = DataLoader(ds_val, batch_size=32, shuffle=False, num_workers=8)
        test_loaders=[]
        for idx in range(len(ds.test)):
            # 1. 确保 MyDataset 实例接收的是单个 DataFrame
            ds_test = WaveDataset(data=ds.test[idx],normalize=True,transform=transforms["val"])
            # 2. 为每个数据集创建独立的 DataLoader
            loader = DataLoader(
                ds_test, 
                batch_size=32, 
                shuffle=False,  # 测试时通常不shuffle
                num_workers=8
            )
            test_loaders.append(loader)

        return dl_train,dl_val,test_loaders
    else:
        ds = WaveDataset(data=ds,normalize=True,transform=transforms["val"])
        dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=8)
        return ds, dl
    
def get_dataset_and_loader_MLAAD(label):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default=f"MultiView/MLAAD_cross_lang")
    parser.add_argument("--test_noise", type=int, default=0)
    parser.add_argument("--gpu", type=int, nargs="+", default=0)
    parser.add_argument("-v", "--version", type=int, default=None)
    args = parser.parse_args()
    cfg = get_cfg_defaults(
        "config/experiments/%s.yaml" % args.cfg, ablation=None
    )
    dataset = MLAAD_AudioDs(root_path=cfg.DATASET.dataset_cfg.root_path)
    if label=="fake":   ds = dataset.get_splits()
    elif label=="real": ds = dataset.get_true_train()
    # elif label=="val": ds = dataset.get_splits_val()
    elif label=="split": ds = dataset.get_splits()
    transforms = build_transforms(cfg.DATASET.transforms, args=args)
    if label=="split":
        ds_train = WaveDataset(data=ds.train,normalize=True,transform=transforms["train"])
        dl_train = DataLoader(ds_train, batch_size=32, shuffle=False, num_workers=8)
        ds_val = WaveDataset(data=ds.val,normalize=True,transform=transforms["val"])
        dl_val = DataLoader(ds_val, batch_size=32, shuffle=False, num_workers=8)
        test_loaders=[]
        for idx in range(len(ds.test)):
            # 1. 确保 MyDataset 实例接收的是单个 DataFrame
            ds_test = WaveDataset(data=ds.test[idx],normalize=True,transform=transforms["val"])
            # 2. 为每个数据集创建独立的 DataLoader
            loader = DataLoader(
                ds_test, 
                batch_size=32, 
                shuffle=False,  # 测试时通常不shuffle
                num_workers=8
            )
            test_loaders.append(loader)

        return dl_train,dl_val,test_loaders
    else:
        ds = WaveDataset(data=ds,normalize=True,transform=transforms["train"])
        dl = DataLoader(ds, batch_size=32, shuffle=False, num_workers=8)
        return ds, dl
    
# ds, dl = get_dataset_and_loader_MLAAD("real")
# print(ds)