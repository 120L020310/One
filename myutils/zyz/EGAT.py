import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_max_pool
from torch_geometric.data import Data, Batch


##降维残差融合方案，参数量少
class EnhancedGAT_reduce(nn.Module):
    def __init__(self, in_dim=512, hidden_dim=256, heads=8, dropout=0.3):
        super().__init__()
        # 第一层GAT：多头注意力
        self.conv1 = GATConv(
            in_dim,
            hidden_dim,
            heads=heads,
            dropout=dropout,
            add_self_loops=True,  # 自动添加自连接
        )
        # 第二层GAT：单头注意力
        self.conv2 = GATConv(
            hidden_dim * heads,
            hidden_dim,  
            heads=1,
            concat=False,
            dropout=dropout,
        )
        
        # 新增的QKV投影层
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)

        # 多尺度融合模块保持原维度
        self.fusion = nn.MultiheadAttention(
            embed_dim=256, num_heads=4, dropout=dropout, batch_first=True
        )
        # 全局池化层
        self.pool = global_max_pool

        # 降维层：将拼接后的2304维降至256
        self.dim_reducer = nn.Linear(hidden_dim * (heads + 1), hidden_dim)

        # 新增分类前的归一化层
        self.norm = nn.LayerNorm(hidden_dim * 2)  # 匹配拼接后的维度
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(256 * 2, 512), nn.ReLU(), nn.Linear(512, 1)  # 256*2=512
        )
        self.bn = nn.BatchNorm1d(hidden_dim * heads)
        self.dropout = nn.Dropout(dropout)
        self.contrib_scorer = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, batch):
        # 第一层GAT
        x1 = self.conv1(x, edge_index)
        x1 = F.leaky_relu(x1)
        x1 = self.bn(x1)
        x1 = self.dropout(x1)

        # 第二层GAT
        x2 = self.conv2(x1, edge_index)  # x2形状 [num_nodes, 256]
        # --- 多尺度特征融合 ---
        # 重组节点特征：每个样本的3个节点
        x1_pooled = x1.view(-1, 3, 2048)  # [batch_size, 3, 2048]
        x2_pooled = x2.view(-1, 3, 256)  # [batch_size, 3, 256]

        # 拼接不同层次的特征
        combined = torch.cat([x1_pooled, x2_pooled], dim=2)  # [batch_size, 3, 2048+256]

        reduced = self.dim_reducer(combined) # [batch_size, 3, 256]

        # 改进的跨节点特征交互
        # 分别计算QKV
        q = self.q_proj(reduced)  # [B, 3, 256]
        k = self.k_proj(reduced)  # [B, 3, 256]
        v = self.v_proj(reduced)  # [B, 3, 256]
        
        # 注意力交互
        fused, _ = self.fusion(q, k, v)
        fused = fused + reduced  # 残差连接

        # 计算每个节点的贡献分数
        contrib_scores = self.contrib_scorer(fused).squeeze(-1)  # [batch_size, 3]
        contrib_weights = F.softmax(contrib_scores, dim=1)  

        # 全局池化 + 拼接
        global_max = fused.max(dim=1)[0]  # [batch_size, 256]
        global_mean = fused.mean(dim=1)  # [batch_size, 256]
        final_features = torch.cat(
            [global_max, global_mean], dim=1
        )  # [batch_size, 512]

        # 添加归一化层
        normalized_features = self.norm(final_features)
        
        # 分类决策
        return self.classifier(normalized_features).squeeze(),contrib_weights,normalized_features


class EnhancedGAT(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=256, heads=8, dropout=0.3):
        super().__init__()
        # 第一层GAT：多头注意力
        self.conv1 = GATConv(
            in_dim,
            hidden_dim,
            heads=heads,
            dropout=dropout,
            add_self_loops=True,  # 自动添加自连接
        )
        # 第二层GAT：单头注意力
        self.conv2 = GATConv(
            hidden_dim * heads,
            hidden_dim,  # 直接映射到单维特征
            heads=1,
            concat=False,
            dropout=dropout,
        )
        # 多尺度融合模块：适配拼接后的特征维度
        self.fusion = nn.MultiheadAttention(
            embed_dim=hidden_dim * (heads + 1),  # 2048+256=2304 → 但需可被num_heads整除
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )
        # 全局池化层
        self.pool = global_max_pool
        # 分类器
        # 分类器适配新维度
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * (heads + 1) * 2, 512),  # 2304*2=4608
            nn.ReLU(),
            nn.Linear(512, 1),
        )
        self.bn = nn.BatchNorm1d(hidden_dim * heads)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, batch):
        # 第一层GAT
        x1 = self.conv1(x, edge_index)
        x1 = F.leaky_relu(x1)
        x1 = self.bn(x1)
        x1 = self.dropout(x1)

        # 第二层GAT
        x2 = self.conv2(x1, edge_index)  # x2形状 [num_nodes, 256]
        # --- 多尺度特征融合 ---
        # 重组节点特征：每个样本的3个节点
        x1_pooled = x1.view(-1, 3, 2048)  # [batch_size, 3, 2048]
        x2_pooled = x2.view(-1, 3, 256)  # [batch_size, 3, 256]

        # 拼接不同层次的特征
        combined = torch.cat([x1_pooled, x2_pooled], dim=2)  # [batch_size, 3, 2048+256]

        # 确保embed_dim可被num_heads整除
        assert (
            self.fusion.embed_dim % self.fusion.num_heads == 0
        ), "embed_dim必须能被num_heads整除"

        # 跨节点特征交互（通过多头注意力）
        fused, _ = self.fusion(combined, combined, combined)  # [batch_size, 3, 2304]

        # 全局池化 + 拼接
        global_max = fused.max(dim=1)[0]  # [batch_size, 2304]
        global_mean = fused.mean(dim=1)  # [batch_size, 2304]
        final_features = torch.cat(
            [global_max, global_mean], dim=1
        )  # [batch_size, 4608]

        # 分类决策
        return self.classifier(final_features).squeeze()


class DynamicGraphBuilder:
    @staticmethod
    def build_batch(features_list, edge_index):
        """
        输入:
            features_list: 包含三个特征的列表，每个特征形状为(batch_size, 1024)
            edge_index: 预先生成的全局边索引
        输出:
            构建好的图batch对象
        """
        data_list = []
        batch_size = features_list[0].size(0)

        for i in range(batch_size):
            x = torch.stack(
                [features_list[0][i], features_list[1][i], features_list[2][i]], dim=0
            )

            data = Data(x=x, edge_index=edge_index.to(x.device))
            data_list.append(data)

        return Batch.from_data_list(data_list)


# 训练流程示例
def train_step(model, batch_data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()

    # 前向传播
    logits = model(batch_data.x, batch_data.edge_index, batch_data.batch)

    # 计算损失
    loss = criterion(logits, batch_data.y.float())

    # 反向传播
    loss.backward()
    optimizer.step()

    return loss.item()


# 使用示例
if __name__ == "__main__":
    # 模拟输入数据
    batch_size = 64
    esr_feat = torch.randn(batch_size, 1024)
    mel_feat = torch.randn(batch_size, 1024)
    wavlm_feat = torch.randn(batch_size, 1024)
    labels = torch.randint(0, 2, (batch_size,))

    # 构建图数据
    graph_builder = DynamicGraphBuilder()
    batch_data = graph_builder.build_batch([esr_feat, mel_feat, wavlm_feat])
    batch_data.y = labels
    batch_data = batch_data.to("cuda")

    # 初始化模型
    model = EnhancedGAT().to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    # 训练步骤
    loss = train_step(model, batch_data, optimizer, criterion)
    print(f"Training loss: {loss:.4f}")
