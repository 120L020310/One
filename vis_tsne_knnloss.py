import argparse
import os
import random
from dataclasses import dataclass
from typing import Optional, Tuple
import warnings

from Oneclass_XLSR_lit_data_aug import ALDA_OneClass_AugImmunity_Lit_detach_noisy

warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tqdm import tqdm

from Oneclass_XLSR_lit_data_aug import ALDA_OneClass_AugImmunity_MultiCenter_Lit
from config.config import get_cfg_defaults
from data.make_dataset import make_data
from myutils.tools._common import to_list

# ====== TODO: 改成你工程里的实际导入 ======
# from your_module import ALDA_OneClass_AugImmunity_MultiCenter_Lit
# 或者：from your_module import ALDA_OneClass_AugImmunity_Lit_detach_noisy
# =======================================


def seed_everything(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def extract_embeddings_from_loader(
    lit_model,
    loader,
    device: str,
    max_per_class: int = 2000,
    max_batches: int = -1,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    抽取测试集 embedding + labels
    - max_per_class: 每个类别最多取多少（real=1, fake=0）
    - normalize: 是否对 final_feat 做 L2 normalize（建议 True）
    返回：
      feats: (N, D) np.float32
      labels: (N,) np.int64  (1=real, 0=fake)
    """
    lit_model.eval()
    lit_model.to(device)

    feats_real, feats_fake = [], []
    count_real, count_fake = 0, 0

    for bi, batch in enumerate(tqdm(loader)):
        if max_batches > 0 and bi >= max_batches:
            break

        audio = batch["audio"].to(device)
        labels = batch["label"].to(device).long()

        if len(audio.shape) == 3:
            audio = audio[:, 0, :]

        # 用和你KNN评估一致的抽特征方式
        res = lit_model.model(audio)
        feat = res["final_feat"]

        if normalize:
            feat = F.normalize(feat, p=2, dim=1)

        feat = feat.detach().cpu().numpy()
        lab = labels.detach().cpu().numpy()

        # 按类收集，做均衡子采样
        for i in range(feat.shape[0]):
            if lab[i] == 1 and count_real < max_per_class:
                feats_real.append(feat[i])
                count_real += 1
            elif lab[i] == 0 and count_fake < max_per_class:
                feats_fake.append(feat[i])
                count_fake += 1

            if count_real >= max_per_class and count_fake >= max_per_class:
                break

        if count_real >= max_per_class and count_fake >= max_per_class:
            break

    feats = np.vstack([np.asarray(feats_real), np.asarray(feats_fake)]).astype(np.float32)
    labels = np.concatenate([
        np.ones(len(feats_real), dtype=np.int64),
        np.zeros(len(feats_fake), dtype=np.int64)
    ])

    print(f"[extract] real={len(feats_real)} fake={len(feats_fake)} dim={feats.shape[1]}")
    return feats, labels


@torch.no_grad()
def extract_centroids_from_ckpt_model(lit_model, device: str, normalize: bool = True) -> Optional[np.ndarray]:
    """
    从 LightningModule 里取中心：
    - 多中心：lit_model.centroids 或 lit_model.loss_fn.centroids
    - 单中心：lit_model.centroid 或 lit_model.loss_fn.centroid

    返回：
      centroids: (K,D) or (1,D) np.float32
      若没有则返回 None
    """
    c = None

    # 先找 module buffer
    if hasattr(lit_model, "centroids") and lit_model.centroids is not None:
        if torch.any(lit_model.centroids != 0):
            c = lit_model.centroids
    elif hasattr(lit_model, "centroid") and lit_model.centroid is not None:
        if torch.any(lit_model.centroid != 0):
            c = lit_model.centroid.unsqueeze(0)

    # 再找 loss_fn 里的 state
    if c is None and hasattr(lit_model, "loss_fn"):
        lf = lit_model.loss_fn
        if hasattr(lf, "centroids") and lf.centroids is not None and torch.any(lf.centroids != 0):
            c = lf.centroids
        elif hasattr(lf, "centroid") and lf.centroid is not None:
            c = lf.centroid.unsqueeze(0)

    if c is None:
        print("[centroids] not found in model/loss_fn.")
        return None

    c = c.to(device)
    if normalize:
        c = F.normalize(c, p=2, dim=1)

    c_np = c.detach().cpu().numpy().astype(np.float32)
    print(f"[centroids] shape={c_np.shape}")
    return c_np


def run_tsne(
    feats: np.ndarray,
    centroids: Optional[np.ndarray],
    seed: int = 0,
    pca_dim: int = 50,
    perplexity: int = 30,
    n_iter: int = 1500,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    对 feats(+centroids) 一起做 PCA->tSNE，保证 centroids 也在同一嵌入空间。
    返回：
      tsne_feats: (N,2)
      tsne_centroids: (K,2) or None
    """
    seed_everything(seed)

    if centroids is not None:
        X = np.vstack([feats, centroids]).astype(np.float32)
        n_cent = centroids.shape[0]
    else:
        X = feats.astype(np.float32)
        n_cent = 0

    # PCA 先降到 50 维（加速+稳定）
    d = X.shape[1]
    pca_dim = min(pca_dim, d)
    Xp = PCA(n_components=pca_dim, random_state=seed).fit_transform(X)

    tsne = TSNE(
        n_components=2,
        random_state=seed,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
    )

    Xt = tsne.fit_transform(Xp)

    tsne_feats = Xt[:-n_cent] if n_cent > 0 else Xt
    tsne_cent = Xt[-n_cent:] if n_cent > 0 else None
    return tsne_feats, tsne_cent


def plot_tsne(
    tsne_feats: np.ndarray,
    labels: np.ndarray,
    tsne_centroids: Optional[np.ndarray],
    title: str,
    outpath: str,
    alpha: float = 0.5,
    s: int = 8,
):
    """
    画图：real vs fake + centroids
    """
    real_mask = labels == 1
    fake_mask = labels == 0

    plt.figure(figsize=(9, 7))
    plt.scatter(tsne_feats[real_mask, 0], tsne_feats[real_mask, 1], s=s, alpha=alpha, label="Real")
    plt.scatter(tsne_feats[fake_mask, 0], tsne_feats[fake_mask, 1], s=s, alpha=alpha, label="Fake")

    if tsne_centroids is not None:
        plt.scatter(tsne_centroids[:, 0], tsne_centroids[:, 1],
                    s=180, marker="X", linewidths=1.2, edgecolors="k", label="Centroids")
        # 标注中心编号
        for i in range(tsne_centroids.shape[0]):
            plt.text(tsne_centroids[i, 0], tsne_centroids[i, 1], str(i),
                     fontsize=10, weight="bold")

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    plt.savefig(outpath, dpi=200)
    plt.close()
    print(f"[plot] saved to: {outpath}")


def visualize_ckpt_tsne(
    ckpt_path: str,
    lit_class,
    test_loader,
    device: str = "cuda",
    max_per_class: int = 300,
    seed: int = 0,
    out_dir: str = "./tsne_out_stage2",
    tag: str = "stage2",
):
    """
    主入口：加载ckpt -> 抽特征 -> tSNE -> 保存图
    """
    # print(f"[load] {ckpt_path}")
    # model = lit_class.load_from_checkpoint(ckpt_path, cfg=None, args=None)

    feats, labels = extract_embeddings_from_loader(
        model, test_loader, device=device, max_per_class=max_per_class, normalize=True
    )
    centroids = extract_centroids_from_ckpt_model(model, device=device, normalize=True)

    # 可选：保存原始数据，后面重复tSNE不必重抽特征
    os.makedirs(out_dir, exist_ok=True)
    npz_path = os.path.join(out_dir, f"{tag}_feats_labels_centroids.npz")
    np.savez(npz_path, feats=feats, labels=labels, centroids=centroids if centroids is not None else np.zeros((0, feats.shape[1]), dtype=np.float32))
    print(f"[save] npz => {npz_path}")

    tsne_feats, tsne_cent = run_tsne(
        feats=feats,
        centroids=centroids,
        seed=seed,
        pca_dim=50,
        perplexity=30,
        n_iter=1500
    )

    out_png = os.path.join(out_dir, f"{tag}_tsne.png")
    plot_tsne(
        tsne_feats=tsne_feats,
        labels=labels,
        tsne_centroids=tsne_cent,
        title=f"t-SNE: Real vs Fake + Centroids ({tag})",
        outpath=out_png,
        alpha=0.55,
        s=9
    )


# =========================
# 用法示例（你要自己接你的 test_loader）
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="GMM")
    parser.add_argument("--dims", type=str, default="[32, 64, 64, 128]")
    parser.add_argument("--ablation", type=str, default=None)
    # parser.add_argument("--specaug", type=str, default='ss')
    parser.add_argument("--gpu", type=int, nargs="+", default=0)
    parser.add_argument("-v", "--version", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_noise", type=int, default=0)

    args = parser.parse_args()

    cfg = get_cfg_defaults(
        "config/experiments/%s.yaml" % args.cfg, ablation=args.ablation
    )
    ds, dl = make_data(cfg.DATASET, args=args)

    CKPT = "/home/zyz/data/dim=128_test/data_aug_SafeRawAugmentor(noise_intensity=0.1, mask_ratio=0.1)_5:1/1228_baseline_multi_centroid_14:52/MultiView/ASV2021_LA/version_0/checkpoints/best-epoch=0-val-eer=0.1225.ckpt"
    DEVICE = "cuda"

    # TODO: 你需要把 lit_class 和 test_loader 换成你的对象
    ckpt_path = "/home/zyz/data/dim=128_test/1227_XLSR_oneclass_aoc/MultiView/ASV2021_LA/version_0/checkpoints/best-epoch=3-val-auc=0.9246.ckpt"
    pack = torch.load("centroids_init_stage2.pt", map_location="cpu")
    centroids_init = pack["centroids"]  # (K,D)
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint["state_dict"]
    state_dict["centroids"] = centroids_init
    # model = ALDA_OneClass_AugImmunity_MultiCenter_Lit()
    model = ALDA_OneClass_AugImmunity_Lit_detach_noisy()
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    # lit_class = ALDA_OneClass_AugImmunity_MultiCenter_Lit()
    # test_loader = ...

    visualize_ckpt_tsne(CKPT, model, to_list(dl.test)[0], device=DEVICE, max_per_class=300, tag="stage2")
