import argparse
import os
import warnings
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.manifold import TSNE
from matplotlib.collections import LineCollection
warnings.filterwarnings("ignore")
from Oneclass_XLSR_lit_data_aug import ALDA_OneClass_AugImmunity_Lit_detach_noisy
from config.config import get_cfg_defaults
from data.make_dataset import make_data
from myutils.tools._common import to_list

# ===== 1) 你的 ckpt 路径 =====
ckpt_path = "/home/zyz/data/dim=128_test/data_aug_SafeRawAugmentor(noise_intensity=0.1, mask_ratio=0.1)_5:1/1227_baseline_detach_noisy/MultiView/ASV2021_LA/version_0/checkpoints/epoch=7-val-auc=0.9602-val-eer=0.0890_no_post_std.ckpt"


# ===== 2) 导入你的模型类（确保此脚本运行环境能 import 到这些定义）=====
# from your_module_path import ALDA_OneClass_AugImmunity_Lit_new


def _ensure_2d_audio(audio: torch.Tensor) -> torch.Tensor:
    # audio: [B, T] or [B, 1, T]
    if audio.ndim == 3:
        audio = audio[:, 0, :]
    return audio


@torch.no_grad()
def extract_embeddings(model, dataloader, device="cuda", max_points_per_class=300):
    """
    返回：
      centroid: [D]
      real_clean, real_aug: [Nr, D]
      fake_clean, fake_aug: [Nf, D]
    并做下采样（每类最多 max_points_per_class 个样本）
    """
    model.eval().to(device)

    real_clean, real_aug = [], []
    fake_clean, fake_aug = [], []

    # 为了“每类最多多少点”，我们用计数截断
    cnt_real = 0
    cnt_fake = 0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Extract embeddings")):
        audio = _ensure_2d_audio(batch["audio"]).to(device)
        label = batch["label"].to(device).long()  # 1=real, 0=fake

        # clean
        res_clean = model.model(audio)
        feat_clean = res_clean["final_feat"]

        # aug
        audio_aug = model.augmentor(audio)
        res_aug = model.model(audio_aug)
        feat_aug = res_aug["final_feat"]

        # normalize
        feat_clean = F.normalize(feat_clean, p=2, dim=1)
        feat_aug = F.normalize(feat_aug, p=2, dim=1)

        # 按 label 分流
        mask_real = (label == 1)
        mask_fake = (label == 0)

        if mask_real.any() and cnt_real < max_points_per_class:
            take = min(int(mask_real.sum().item()), max_points_per_class - cnt_real)
            idxs = torch.where(mask_real)[0][:take]
            real_clean.append(feat_clean[idxs].detach().cpu())
            real_aug.append(feat_aug[idxs].detach().cpu())
            cnt_real += take

        if mask_fake.any() and cnt_fake < max_points_per_class:
            take = min(int(mask_fake.sum().item()), max_points_per_class - cnt_fake)
            idxs = torch.where(mask_fake)[0][:take]
            fake_clean.append(feat_clean[idxs].detach().cpu())
            fake_aug.append(feat_aug[idxs].detach().cpu())
            cnt_fake += take

        if cnt_real >= max_points_per_class and cnt_fake >= max_points_per_class:
            break

    real_clean = torch.cat(real_clean, dim=0) if len(real_clean) else torch.empty(0)
    real_aug   = torch.cat(real_aug, dim=0)   if len(real_aug) else torch.empty(0)
    fake_clean = torch.cat(fake_clean, dim=0) if len(fake_clean) else torch.empty(0)
    fake_aug   = torch.cat(fake_aug, dim=0)   if len(fake_aug) else torch.empty(0)

    centroid = model.centroid.detach().cpu()

    centroid = F.normalize(centroid, p=2, dim=0)

    return centroid, real_clean, real_aug, fake_clean, fake_aug


def compute_immunity_metrics(centroid, real_clean, real_aug, fake_clean, fake_aug):
    """
    打印两类指标：
    1) clean-aug cosine similarity（越高越免疫）
    2) centroid similarity change: sim(clean, c) vs sim(aug, c) 的差值（绝对值越小越稳）
    """
    def cos_pair(a, b):
        return (a * b).sum(dim=1)

    def cos_to_centroid(x, c):
        c = c.unsqueeze(0).expand_as(x)
        return (x * c).sum(dim=1)

    out = {}

    if real_clean.numel() > 0:
        sim_ra = cos_pair(real_clean, real_aug).numpy()
        sim_rc = cos_to_centroid(real_clean, centroid).numpy()
        sim_rac = cos_to_centroid(real_aug, centroid).numpy()
        delta_r = sim_rac - sim_rc
        out["real_clean_aug_cos_mean"] = float(sim_ra.mean())
        out["real_clean_aug_cos_std"]  = float(sim_ra.std())

        out["real_clean_centroid_cos_mean"] = float(sim_rc.mean())
        out["real_aug_centroid_cos_mean"]   = float(sim_rac.mean())
        out["real_aug_centroid_cos_std"]    = float(sim_rac.std())

        out["real_centroid_delta_mean"] = float(delta_r.mean())
        out["real_centroid_delta_abs_mean"] = float(np.abs(delta_r).mean())

    if fake_clean.numel() > 0:
        sim_fa = cos_pair(fake_clean, fake_aug).numpy()
        sim_fc = cos_to_centroid(fake_clean, centroid).numpy()
        sim_fac = cos_to_centroid(fake_aug, centroid).numpy()
        delta_f = sim_fac - sim_fc
        out["fake_clean_aug_cos_mean"] = float(sim_fa.mean())
        out["fake_clean_aug_cos_std"]  = float(sim_fa.std())

        out["fake_clean_centroid_cos_mean"] = float(sim_fc.mean())
        out["fake_aug_centroid_cos_mean"]   = float(sim_fac.mean())
        out["fake_aug_centroid_cos_std"]    = float(sim_fac.std())

        out["fake_centroid_delta_mean"] = float(delta_f.mean())
        out["fake_centroid_delta_abs_mean"] = float(np.abs(delta_f).mean())

    print("\n===== Immunity metrics =====")
    for k, v in out.items():
        print(f"{k}: {v:.6f}")

    # 一个简单结论提示
    if ("real_clean_aug_cos_mean" in out) and ("fake_clean_aug_cos_mean" in out):
        print("\n[Hint]")
        print("如果 real_clean_aug_cos_mean 明显 > fake_clean_aug_cos_mean，说明 real 对 augmentor 更稳定/免疫。")
    return out


def plot_tsne_with_links(centroid, real_clean, real_aug, fake_clean, fake_aug,
                         perplexity=30, random_state=0, save_path=None):
    """
    画 t-SNE：
      - centroid 单点
      - real clean / aug（线连接）
      - fake clean / aug（线连接）
    """
    # 拼接全部点做 t-SNE
    # 顺序： centroid, real_clean, real_aug, fake_clean, fake_aug
    X = torch.cat([
        centroid.unsqueeze(0),
        real_clean, real_aug,
        fake_clean, fake_aug
    ], dim=0).numpy()

    # t-SNE
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=random_state
    )
    Z = tsne.fit_transform(X)

    # 切片索引
    idx = 0
    z_centroid = Z[idx:idx+1]; idx += 1
    n_real = real_clean.shape[0]
    n_fake = fake_clean.shape[0]

    z_real_clean = Z[idx:idx+n_real]; idx += n_real
    z_real_aug   = Z[idx:idx+n_real]; idx += n_real
    z_fake_clean = Z[idx:idx+n_fake]; idx += n_fake
    z_fake_aug   = Z[idx:idx+n_fake]; idx += n_fake

    # 线段（clean->aug）
    real_segments = np.stack([z_real_clean, z_real_aug], axis=1) if n_real > 0 else None
    fake_segments = np.stack([z_fake_clean, z_fake_aug], axis=1) if n_fake > 0 else None

    plt.figure(figsize=(12, 9))

    # 线：先画线再画点，更清楚
    if real_segments is not None:
        lc = LineCollection(real_segments, linewidths=0.6, alpha=0.25,colors="tab:blue")
        plt.gca().add_collection(lc)
    if fake_segments is not None:
        lc = LineCollection(fake_segments, linewidths=0.6, alpha=0.25,colors="tab:orange")
        plt.gca().add_collection(lc)

    # 点
    if n_real > 0:
        plt.scatter(z_real_clean[:, 0], z_real_clean[:, 1], s=18, alpha=0.75, marker="o", label="Real (clean)")
        plt.scatter(z_real_aug[:, 0],   z_real_aug[:, 1],   s=18, alpha=0.75, marker="x", label="Real (aug)")
    if n_fake > 0:
        plt.scatter(z_fake_clean[:, 0], z_fake_clean[:, 1], s=18, alpha=0.75, marker="o", label="Fake (clean)")
        plt.scatter(z_fake_aug[:, 0],   z_fake_aug[:, 1],   s=18, alpha=0.75, marker="x", label="Fake (aug)")

    # centroid
    plt.scatter(z_centroid[:, 0], z_centroid[:, 1], s=220, marker="*", label="Centroid")

    plt.title("t-SNE: Clean vs Aug Embeddings (with centroid and clean→aug links)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        print(f"\nSaved t-SNE figure to: {save_path}")

    plt.show()


def main(dataloader,idx):
    # ===== 3) 加载模型 =====
    # Lightning: 推荐 strict=False，避免你代码改动导致的 key 不一致
    model = ALDA_OneClass_AugImmunity_Lit_detach_noisy.load_from_checkpoint(
        ckpt_path,
        map_location="cpu",
        strict=True
    )

    # 关键：把 checkpoint 里的 buffer centroid 同步给 loss_fn.centroid（否则你 loss_fn 内部可能还是 init 的值）
    if getattr(model, "loss_fn", None) is not None:
        model.loss_fn.centroid = model.centroid.clone().detach()

    device = "cuda:1" if torch.cuda.is_available() else "cpu"

    # ===== 4) 提取 embeddings =====
    centroid, real_clean, real_aug, fake_clean, fake_aug = extract_embeddings(
        model=model,
        dataloader=dataloader,
        device=device,
        max_points_per_class=300,   # 你可以调大/调小
    )

    # ===== 5) 计算免疫性指标 =====
    compute_immunity_metrics(centroid, real_clean, real_aug, fake_clean, fake_aug)

    # ===== 6) t-SNE 可视化 =====
    plot_tsne_with_links(
        centroid=centroid,
        real_clean=real_clean,
        real_aug=real_aug,
        fake_clean=fake_clean,
        fake_aug=fake_aug,
        perplexity=30,
        random_state=0,
        save_path=f"./tsne_aug_immunity[{idx}].png"
    )


# =====================
# 你只需要像这样调用：
# =====================
# main(dataloader)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="GMM")
    parser.add_argument("--dims", type=str, default="[32, 64, 64, 128]")
    parser.add_argument("--nblocks", type=str, default="[1,1,3,1]")
    parser.add_argument("--ablation", type=str, default=None)

    # parser.add_argument("--specaug", type=str, default='ss')
    parser.add_argument("--gpu", type=int, nargs="+", default=0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--grad", type=int, default=1)
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--earlystop", type=int, default=6)
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
    parser.add_argument("--ckpt_saved_filename", type=str,default="best-{epoch}-{val-eer:.4f}")
    parser.add_argument("--ckpt_train_model_task",type=str,default=None)
    parser.add_argument("--loss_fn",type=str,default="all_loss")
    parser.add_argument("--root_dir",type=str,default="/home/zyz/data/1224_GPT_methods/data_aug_SafeRawAugmentor(noise_intensity=0.1, mask_ratio=0.1)_5:1/1226_baseline")
    args = parser.parse_args()
    # args.gpu=[0,1]

    cfg = get_cfg_defaults(
        "config/experiments/%s.yaml" % args.cfg, ablation=args.ablation
    )
    if args.batch_size > 0:
        cfg.DATASET.test_batch_size = args.batch_size
    ds, dl = make_data(cfg.DATASET, args=args)
    idx = 0
    for test_dl in to_list(dl.test)[:2]:
        main(dataloader=test_dl, idx=idx)
        idx+=1