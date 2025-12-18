import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output, labels):
        """
        对比损失函数：真实音频尽量接近，伪造音频尽量远离
        :param output: GAT 网络的输出，形状为 [192, 64]
        :param labels: 每个音频的标签，形状为 [64]，1 表示真实音频，0 表示伪造音频
        :return: 损失值
        """
        batch_size = output.size(0) // 3  # 假设有三个模态
        audio_features = output[0:batch_size]  # 64个音频特征，形状为 [64, 64]
        asr_features = output[
            batch_size : batch_size + batch_size
        ]  # 64个ASR特征，形状为 [64, 64]
        spec_features = output[
            batch_size * 2 : batch_size * 3
        ]  # 64个梅尔频谱特征，形状为 [64, 64]

        loss = 0.0

        for i in range(batch_size):
            # 获取每个音频的三个特征
            features = [
                audio_features[i],
                asr_features[i],
                spec_features[i],
            ]  # 3个特征，维度为 (3, 64)

            if labels[i] == 1:  # 真实音频，距离应尽量小
                # 计算真实音频的三个特征之间的距离
                for j in range(3):
                    for k in range(j + 1, 3):
                        dist = F.pairwise_distance(features[j], features[k], p=2)
                        loss += dist  # 真实音频距离越小越好，增加损失
            else:  # 伪造音频，某两个特征之间的距离应尽量大
                # 计算伪造音频的两个特征之间的最大距离
                distances = []
                for j in range(3):
                    for k in range(j + 1, 3):
                        dist = F.pairwise_distance(features[j], features[k], p=2)
                        distances.append(dist)

                max_distance = torch.max(torch.stack(distances))  # 找到最大距离
                loss += torch.max(
                    torch.tensor(0.0, device=distances[0].device),
                    self.margin - max_distance,
                )  # 如果最大距离小于margin，增加损失

        return loss
