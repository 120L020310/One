import argparse
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch
from tqdm import tqdm
import torch.nn.functional as F

from Oneclass_XLSR_lit import ALDA_OneClass_Lit
from config.config import get_cfg_defaults
from data.make_dataset import make_data
from get_true_dataset import get_true
from vis_tsne_oneclass import compute_eer
def extract_bank(model,train_loader,device,memory_bank_path, k=5):
    model.eval()
    print("Building k-NN Memory Bank from Training Data...")
    memory_bank = []
    with torch.no_grad():
        for batch in tqdm(train_loader, desc="Extracting Train Feats"):
            # 只需要 Real 样本 (Label=1)
            audio = batch["audio"][batch["label"]==1].to(device)
            if len(audio) == 0: continue
            
            if len(audio.shape) == 3: audio = audio[:, 0, :]
            
            res = model.model(audio)
            feat = F.normalize(res["final_feat"], p=2, dim=1)
            memory_bank.append(feat.cpu().numpy())
            
        memory_bank = np.concatenate(memory_bank, axis=0)
        print(f"Memory Bank Size: {memory_bank.shape}") # (N_train_real, 1024)
        torch.save(memory_bank,memory_bank_path)
        # 训练 k-NN 索引
def evaluate_knn(model, memory_bank, test_loader, device, k=5):
   # -------------------------------------------
   # 2. 测试阶段 (k-NN Scoring)
   # -------------------------------------------
    print("Evaluating Test Data with k-NN...")
    all_scores = []
    all_labels = []
    knn = NearestNeighbors(n_neighbors=k, metric="cosine", n_jobs=-1)
    knn.fit(memory_bank)
    with torch.no_grad():
        idx = 0
        for batch in tqdm(test_loader, desc="Testing"):
            idx+=1
            audio = batch["audio"].to(device)
            label = batch["label"]

            if len(audio.shape) == 3: audio = audio[:, 0, :]

            res = model.model(audio)
            feat_test = F.normalize(res["final_feat"], p=2, dim=1) # (B, 256)

            # 查询最近的 k 个邻居
            # distances: (B, k), indices: (B, k)
            dists, _ = knn.kneighbors(feat_test.cpu().numpy())

            # Anomaly Score = 平均距离
            # 距离越远，越可能是 Fake
            knn_score = dists.mean(axis=1)

            all_scores.extend(knn_score)
            all_labels.extend(label.cpu().numpy())
            if idx==50:break
    # -------------------------------------------
    # 3. 计算指标
    # -------------------------------------------
    y_true_anomaly = 1 - np.array(all_labels) # Real=0, Fake=1
    scores = np.array(all_scores)
   
    eer, auc_score = compute_eer(y_true_anomaly, scores)
    
    print(f"\n[k-NN Result (k={k})]")
    print(f"  > AUC : {auc_score*100:.2f}%")
    print(f"  > EER : {eer*100:.2f}%")
   
    return eer, auc_score
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="GMM")
    parser.add_argument("--gpu", type=int, nargs="+", default=0)
    parser.add_argument("-v", "--version", type=int, default=None)
    parser.add_argument("--test_noise", type=int, default=0)
    parser.add_argument("-nr","--noise_ratio",type=float,default=0.5) 
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("-ckpt", "--checkpoint", type=str,default="/home/zyz/data/test_controlled_ex/1221_XLSR_oneclass_aoc_tight/MultiView/ASV2021_LA/version_0/checkpoints/best-epoch=19-val-auc=0.9267.ckpt")
    args = parser.parse_args()
    
    # 1. 配置环境
    cfg = get_cfg_defaults("config/experiments/%s.yaml" % args.cfg)
    cfg.DATASET.batch_size = args.batch_size
    device = torch.device(f'cuda:{args.gpu[0]}' if torch.cuda.is_available() else 'cpu')
    
    # 2. 加载模型
    print(f"Loading model from {args.checkpoint} ...")
    model = ALDA_OneClass_Lit()
    
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    sd = checkpoint["state_dict"]
    model.load_state_dict(sd, strict=False)
    model = model.to(device)
    model.eval()
    ds, dl = make_data(cfg.DATASET, args=args)
    train_ds, train_dl = get_true(["ASV2021_inner","ASV2021_LA"],"val")
    test_dataloaders = dl.test if isinstance(dl.test, list) else [dl.test]
    memory_bank_path = "memory_bank_xlsr_best_tight_aoc.pt"
    extract_bank(model,train_dl,device,memory_bank_path)
    # 5. 遍历测试集
    memory_bank = torch.load("memory_bank_xlsr_best_tight_aoc.pt")
    for i, test_loader in enumerate(test_dataloaders):
        evaluate_knn(model,memory_bank,test_loader,device)