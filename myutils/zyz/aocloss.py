import torch
import torch.nn as nn
from torch.autograd.function import Function
import torch.nn.functional as F
from torch.autograd import Variable
import math
class AOCloss(nn.Module):
    '''
    AOC loss function.
    based on the paper:
    Adaptive Centroid Shift Loss (AOCloss) method for Audio Deepfake Detection
    '''
    def __init__(self, embedding_dim=2):
        super(AOCloss, self).__init__()
        self.embedding_dim = embedding_dim
        self.centroid = None
        self.n = 0  # Total number of bonafide samples encountered

    def update_centroid(self, bonafide_embeddings):
        s = bonafide_embeddings.shape[0]
        if s == 0:
            return

        Ei = bonafide_embeddings.mean(dim=0).detach()  # Detach to avoid graph tracking

        if self.centroid is None:
            self.centroid = Ei
            self.n = s
        else:
            self.centroid = ((self.n * self.centroid.detach()) + (s * Ei)) / (self.n + s)
            self.n += s

    def one_class_loss(self, bonafide_embeddings, fake_embeddings):
        if self.centroid is None:
            raise ValueError("Centroid has not been initialized with bonafide samples.")

        # Normalize embeddings and centroid
        centroid_norm = F.normalize(self.centroid.detach(), p=2, dim=0)
        bonafide_norm = F.normalize(bonafide_embeddings, p=2, dim=1)
        fake_norm = F.normalize(fake_embeddings, p=2, dim=1)

        # Compute cosine similarity
        bonafide_similarity = torch.matmul(bonafide_norm, centroid_norm)
        fake_similarity = torch.matmul(fake_norm, centroid_norm)

        Mb = bonafide_embeddings.shape[0]
        Ms = fake_embeddings.shape[0]

        if Mb == 0 :
            loc = 1 + (torch.sum(fake_similarity) / Ms)
        elif Ms ==0:
            loc = 1 + (-torch.sum(bonafide_similarity) / Mb)
            # raise ValueError("Both bonafide and spoof samples must be present in the batch.")
        else:
            loc = 1 + (-torch.sum(bonafide_similarity) / Mb) + (torch.sum(fake_similarity) / Ms)
        return loc

    def forward(self, embeddings, labels=None,stage = "train"):
        bonafide_embeddings = embeddings[labels == 1]
        fake_embeddings = embeddings[labels == 0]
        # if stage =="train":
        self.update_centroid(bonafide_embeddings)
        loss = self.one_class_loss(bonafide_embeddings, fake_embeddings)
        return loss

class AOCloss_plus(nn.Module):
    def __init__(self, embedding_dim=256, momentum=0.9):
        super(AOCloss_plus, self).__init__()
        self.embedding_dim = embedding_dim
        self.centroid = None
        self.initialized = False
        self.momentum = momentum # 动量系数，越小更新越快，0.9 比较平滑

    def update_centroid(self, bonafide_embeddings):
        # 此时 bonafide_embeddings 已经是 (Batch, Dim)
        if bonafide_embeddings.shape[0] == 0:
            return

        # 计算当前 Batch 的中心
        current_batch_center = bonafide_embeddings.mean(dim=0).detach()

        if not self.initialized or self.centroid.sum() == 0:
            self.centroid = current_batch_center
            self.initialized = True
        else:
            # EMA 更新公式： Old * m + New * (1-m)
            # 这样 Centroid 永远紧跟最近几个 Batch 的特征分布，不会被几十个 Epoch 前的旧参数拖累
            self.centroid = (self.momentum * self.centroid) + \
                            ((1 - self.momentum) * current_batch_center)

    def forward(self, embeddings, labels=None, stage="train"):
        # 确保 embeddings 已经做过 L2 Normalize (在外部做)
        # 或者在这里加: embeddings = F.normalize(embeddings, p=2, dim=1)
        
        bonafide_embeddings = embeddings[labels == 1]
        fake_embeddings = embeddings[labels == 0]

        if stage == "train" and bonafide_embeddings.shape[0] > 0:
            self.update_centroid(bonafide_embeddings)
        # self.update_centroid(bonafide_embeddings)

        # 归一化 Centroid
        centroid_norm = F.normalize(self.centroid, p=2, dim=0)
        
        # 计算 Loss
        # 训练时我们假设全是 bonafide (根据你的逻辑)
        # Loss = 1 - CosSim(x, c)
        if bonafide_embeddings.shape[0] > 0:
            # 这里的逻辑是：让 Bonafide 靠近中心
            # 1 - mean(sim) 等价于 mean(dist)
            bonafide_sim = torch.matmul(bonafide_embeddings, centroid_norm)
            loss = 1 - bonafide_sim.mean()
        else:
            loss = torch.tensor(0.0, device=embeddings.device, requires_grad=True)
            
        return loss
def infer_one_class(DNN_model, new_samples):
    """
    Perform inference using a trained one-class ACS model for a batch of samples.

    Args:
        DNN_model (torch.nn.Module): The trained model with an updated centroid.
        new_samples (torch.Tensor): The input batch of samples for inference.
        threshold (float): The decision threshold for classification.

    Returns:
        list: Classification results ('Bonafide' or 'Spoof') for each sample.
        list: Computed cosine similarity scores for each sample.
    """
    DNN_model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    similarities = []
    
    with torch.no_grad():
        # Move samples to device
        new_samples = new_samples.to(device)
        
        # Compute embeddings
        embeddings = DNN_model(new_samples.float())
        
        # Normalize embeddings and centroid
        embeddings = F.normalize(embeddings, p=2, dim=1)
        centroid = F.normalize(DNN_model.loss.centroid, p=2, dim=0)
        
        # Calculate cosine similarities for each sample
        similarities = torch.matmul(embeddings, centroid).cpu().numpy()
       
    return similarities


if __name__ == "__main__":
    # Example usage:
    embedding_dim = 512
    criterion  = AOCloss(embedding_dim=embedding_dim)

    # Generate random embeddings and corresponding labels (0: bonafide, 1: spoof)
    embeddings = torch.randn(100, embedding_dim)
    labels = torch.cat([torch.zeros(10, dtype=torch.int), torch.ones(90, dtype=torch.int)])

    # Calculate loss
    loss = criterion(embeddings, labels)
    print("Loss:", loss.item())

