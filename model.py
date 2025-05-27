import torch
import torch.nn as nn

class PANet(nn.Module):
    def __init__(self, num_classes=6, embed_dim=256, num_heads=4, dropout_rate=0.1):
        super().__init__()
        self.embed_dim = embed_dim

        self.pointwise_mlp = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
            nn.ReLU()
        )

        self.Multihead_attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)

        self.dropout = nn.Dropout(dropout_rate)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        feat = self.pointwise_mlp(x)

        attn_out, _ = self.Multihead_attn(feat, feat, feat)
        feat = self.dropout(attn_out + feat)

        max_pool = torch.max(feat, dim=1)[0]
        mean_pool = torch.mean(feat, dim=1)
        feat = torch.cat([max_pool, mean_pool], dim=1)

        y = self.mlp(feat)
        return y