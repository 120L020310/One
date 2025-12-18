import argparse

import pytorch_lightning as pl
import torch
import warnings

from controlled_ex.Tinyvit.lit_model import TinyVit_lit
from controlled_ex.Tinyvit.model import TinyVit

warnings.filterwarnings("ignore")

pl.seed_everything(42)
torch.set_float32_matmul_precision("medium")
torch.backends.cudnn.benchmark = True

from controlled_ex.Hubert.lit_model import Hubert_lit
from controlled_ex.LibriSeVec.litmodel import LibriSeVoc_lit
from myutils.tools import color_print, to_list

from config import get_cfg_defaults
from data.make_dataset import make_data
from controlled_ex.Aaasist.lit_aaasist import AASIST_lit
from utils import (
    # clear_folder,
    build_logger,
    get_ckpt_path,
    make_callbacks,
    write_model_summary,
)

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# ROOT_DIR = "/home/zyz/data/test_controlled_ex/Aaasist_baseline"

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="GMM")
    parser.add_argument("--dims", type=str, default="[32, 64, 64, 128]")
    parser.add_argument("--nblocks", type=str, default="[1,1,3,1]")
    parser.add_argument("--ablation", type=str, default=None)

    # parser.add_argument("--specaug", type=str, default='ss')
    parser.add_argument("--gpu", type=int, nargs="+", default=0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--grad", type=int, default=1)
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--earlystop", type=int, default=3)
    parser.add_argument("--min_epoch", type=int, default=1)
    parser.add_argument("--use_profiler", type=int, default=0)
    parser.add_argument("--use_lr_find", type=int, default=0)
    parser.add_argument("-v", "--version", type=int, default=None)
    parser.add_argument("-t", "--test", type=int, default=0)
    parser.add_argument("-l", "--log", type=int, default=0)
    parser.add_argument("--resume", type=int, default=0)
    parser.add_argument("--theme", type=str, default="best")
    parser.add_argument("--collect", type=int, default=0)
    parser.add_argument("--clear_log", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_as_val", type=int, default=999)
    parser.add_argument("--test_noise", type=int, default=0)
    parser.add_argument("--test_noise_level", type=int, default=30)
    parser.add_argument("--test_noise_type", type=str, default="bg")
    parser.add_argument("--ckpt_saved_filename", type=str,default="best-{epoch}-{val-auc:.4f}")
    parser.add_argument("--ckpt_train_model_task",type=str,default=None)
    parser.add_argument("--loss_fn",type=str,default="all_loss")
    parser.add_argument("--root_dir",type=str,default="/home/zyz/data/work2/test_controlled_ex/TinyVit2019LA_baseline")
    args = parser.parse_args()
    # args.gpu=[0,1]
    if args.seed != 42:
        pl.seed_everything(args.seed)

    cfg = get_cfg_defaults(
        "config/experiments/%s.yaml" % args.cfg, ablation=args.ablation
    )
    if args.batch_size > 0:
        cfg.DATASET.batch_size = args.batch_size
    ds, dl = make_data(cfg.DATASET, args=args)
    
    args.profiler = (
        pl.profilers.SimpleProfiler(dirpath="./", filename="test")
        if args.use_profiler
        else None
    )

    # print(str(dict(cfg)))
    model = TinyVit_lit()
    callbacks = make_callbacks(args, cfg)

    if args.ckpt_saved_filename:
        callbacks[5].filename=args.ckpt_saved_filename

    logger = build_logger(args, args.root_dir)

    trainer = pl.Trainer(
        max_epochs=2 if args.use_profiler else cfg.MODEL.epochs,
        accelerator="gpu",
        devices=args.gpu,
        logger=logger,
        check_val_every_n_epoch=1,
        callbacks=callbacks,
        default_root_dir=args.root_dir,
        strategy="ddp_find_unused_parameters_true" if len(args.gpu) > 1 else "auto",
        profiler=args.profiler,
        enable_checkpointing=False if args.use_profiler else True,
        limit_train_batches=100 if args.use_profiler else 1.0,
        limit_val_batches=100 if args.use_profiler else 1.0,
        num_sanity_val_steps=0,
        accumulate_grad_batches=args.grad,
        precision="16-mixed" if args.precision == 16 else "32",
    )
    log_dir = trainer.logger.log_dir
    # log_dir = '/home/zyz/data/test_model_save/1-df-audio/MultiView/ASV2019_LA/version_1'
    color_print(f"logger path : {log_dir}")
    

    # if args.test == 0 and args.clear_log:
    #     clear_folder(log_dir)
    # args.test = True
    if not args.test:
        ckpt_path = get_ckpt_path(log_dir, theme="last") if args.resume else None

        # val_dl = to_list(dl.test)[1]
        val_dl = dl.val
        if args.test_as_val != 999:
            val_dl = to_list(dl.test)[args.test_as_val]

        trainer.fit(model, dl.train, val_dataloaders=val_dl, ckpt_path=ckpt_path)

        write_model_summary(model, log_dir)
        ckpt_path = get_ckpt_path(log_dir, theme=args.theme)
        trainer.trainset_wo_transform = dl.train_wo_transform

        if not args.collect:
            for test_dl in to_list(dl.test):
                trainer.test(model, test_dl, ckpt_path=ckpt_path)
    else:
        ckpt_path = get_ckpt_path(log_dir, theme=args.theme)
        trainer.trainset_wo_transform = dl.train_wo_transform

        if not args.collect:
            for test_dl in to_list(dl.test):
                trainer.test(model, test_dl, ckpt_path=ckpt_path)
        else:
            for test_dl in to_list(dl.test) + [dl.val]:
                trainer.test(model, test_dl, ckpt_path=ckpt_path)
