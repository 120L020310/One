import warnings
import torch
import torch.nn.functional as F
from tqdm import tqdm

from Oneclass_XLSR_lit import XLSR_OneClass_Lit
from Oneclass_XLSR_lit_data_aug_new_detach_noisy import ALDA_OneClass_AugImmunity_Lit_detach_noisy
from get_true_dataset import get_true

warnings.filterwarnings("ignore")

@torch.no_grad()
def build_memory_bank(
    lit_model,
    train_loader,
    device,
    max_batches: int = -1,
) -> torch.Tensor:
    """
    Extract normalized embeddings on TRAIN REAL set using stage1 ckpt.
    Returns:
      memory_bank: (N, D) torch tensor on CPU
    """
    lit_model.eval()
    lit_model.to(device)

    feats = []
    for bi, batch in enumerate(tqdm(train_loader)):
        if max_batches > 0 and bi >= max_batches:
            break
        audio = batch["audio"].to(device)
        if len(audio.shape) == 3:
            audio = audio[:, 0, :]

        res = lit_model.model(audio)
        z = res["final_feat"]
        z = F.normalize(z, p=2, dim=1)
        feats.append(z.detach().cpu())

    return torch.cat(feats, dim=0)


@torch.no_grad()
def torch_kmeans(
    x: torch.Tensor,          # (N,D) on CPU or GPU
    k: int,
    iters: int = 50,
    tol: float = 1e-4,
    seed: int = 0,
) -> torch.Tensor:
    """
    Simple cosine kmeans (spherical kmeans style):
      - normalize x
      - assignment by max cosine
      - centroid update = mean then normalize
    Returns:
      centroids: (K, D)
    """
    g = torch.Generator(device=x.device)
    g.manual_seed(seed)

    x = F.normalize(x, p=2, dim=1)

    N, D = x.shape
    # init: random sample
    idx = torch.randperm(N, generator=g, device=x.device)[:k]
    c = x[idx].clone()  # (K,D)

    prev_obj = None
    for _ in range(iters):
        sims = x @ c.t()                # (N,K)
        assign = sims.argmax(dim=1)     # (N,)
        sim_max = sims.max(dim=1).values
        obj = 1.0 - sim_max.mean()      # smaller is better

        # update
        c_new = torch.zeros_like(c)
        for kk in range(k):
            mask = (assign == kk)
            if mask.any():
                c_new[kk] = x[mask].mean(dim=0)
            else:
                # re-init empty cluster
                ridx = torch.randint(0, N, (1,), generator=g, device=x.device).item()
                c_new[kk] = x[ridx]
        c_new = F.normalize(c_new, p=2, dim=1)

        if prev_obj is not None and abs(prev_obj - float(obj.item())) < tol:
            c = c_new
            break
        c = c_new
        prev_obj = float(obj.item())

    return c


device = "cuda"
stage1 = XLSR_OneClass_Lit.load_from_checkpoint("/home/zyz/data/dim=128_test/1227_XLSR_oneclass_aoc/MultiView/ASV2021_LA/version_0/checkpoints/best-epoch=3-val-auc=0.9246.ckpt")
# stage2 = ALDA_OneClass_AugImmunity_Lit_detach_noisy.load_from_checkpoint("/home/zyz/data/dim=128_test/data_aug_SafeRawAugmentor(noise_intensity=0.1, mask_ratio=0.1)_5:1/1227_baseline_detach_noisy/MultiView/ASV2021_LA/version_0/checkpoints/epoch=7-val-auc=0.9602-val-eer=0.0890_no_post_std.ckpt")
train_ds, train_dl = get_true(["ASV2021_inner","ASV2021_LA"],"val",bs=16) 
memory_bank = build_memory_bank(stage1, train_dl, device=device, max_batches=-1)  # (N,D) 
K = 8
centroids_init = torch_kmeans(memory_bank.to(device), k=K, iters=50).detach().cpu()  # (K,D) CPU
torch.save({"centroids": centroids_init}, "centroids_init_stage2.pt")
