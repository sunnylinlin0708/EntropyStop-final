import torch
import torch.nn as nn

# 原本使用 torch_geometric.nn.MLP([in_dim, 64, in_dim], dropout=0.2)。
# 因 PyG 的相依套件（xxhash）的 DLL 被 Windows 應用程式控制原則封鎖，
# 改用等效的純 PyTorch 寫法。PyG MLP 的預設行為為：
#   Linear -> BatchNorm -> ReLU -> Dropout(0.2) -> Linear（最後一層不加正規化、激活與 dropout）
class Autoencoder(nn.Module):
    def __init__(self, in_dim):
        super(Autoencoder, self).__init__()
        h_dim = 64
        self.lins = nn.Sequential(
            nn.Linear(in_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(h_dim, in_dim),
        )

    def forward(self, x):
        x_ = self.lins(x)
        return torch.sum(torch.square(x_ - x), dim=-1)