import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        HIDDEN_DIM = 32
        self.net = nn.Sequential(
            nn.Conv1d(3, HIDDEN_DIM, 25, 2, 12),
            nn.ReLU(),
            nn.Conv1d(HIDDEN_DIM, HIDDEN_DIM, 25, 2, 12),
            nn.ReLU(),
            nn.Conv1d(HIDDEN_DIM, HIDDEN_DIM, 25, 2, 12),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)

class TimeAttentionMIL(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        HIDDEN_DIM = 32
        DROPOUT = 0.3
        self.encoder = encoder
        self.attn = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        self.mlp = nn.Sequential(
            nn.Linear(HIDDEN_DIM, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT),
            nn.Linear(64, 1)
        )

    def forward(self, bag):
        B, N, C, L = bag.shape
        bag = bag.view(B * N, C, L)
        feat = self.encoder(bag).mean(dim=2).view(B, N, -1)
        attn_scores = self.attn(feat).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=1)
        bag_repr = torch.sum(attn_weights.unsqueeze(-1) * feat, dim=1)
        out = self.mlp(bag_repr).squeeze(-1)
        return out, attn_weights
