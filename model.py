import torch
import torch.nn as nn

class FeatureTokenizer(nn.Module):
    def __init__(self, num_features, embed_dim):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_features, embed_dim) * 0.02)
        self.bias = nn.Parameter(torch.zeros(num_features, embed_dim))
        self.feature_embedding = nn.Parameter(torch.randn(num_features, embed_dim) * 0.02)

    def forward(self, x):
        x = x.unsqueeze(-1) * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)
        return x + self.feature_embedding.unsqueeze(0)


class SAINTBlock(nn.Module):
    def __init__(self, embed_dim=32, num_heads=4, dropout=0.1):
        super().__init__()
        self.feature_norm = nn.LayerNorm(embed_dim)
        self.feature_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.feature_dropout = nn.Dropout(dropout)
        self.feature_ffn_norm = nn.LayerNorm(embed_dim)
        self.feature_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout)
        )
        self.sample_norm = nn.LayerNorm(embed_dim)
        self.sample_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.sample_dropout = nn.Dropout(dropout)
        self.sample_ffn_norm = nn.LayerNorm(embed_dim)
        self.sample_ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x_norm = self.feature_norm(x)
        x_attn, _ = self.feature_attention(x_norm, x_norm, x_norm)
        x = x + self.feature_dropout(x_attn)
        x = x + self.feature_ffn(self.feature_ffn_norm(x))

        sample_repr = x.mean(dim=1, keepdim=True)
        sample_norm = self.sample_norm(sample_repr)
        sample_attn, _ = self.sample_attention(sample_norm, sample_norm, sample_norm)
        sample_repr = sample_repr + self.sample_dropout(sample_attn)
        sample_repr = sample_repr + self.sample_ffn(self.sample_ffn_norm(sample_repr))

        x = x + sample_repr
        return x


class DLASS_PRP(nn.Module):
    def __init__(self, num_features=13, num_classes=3, embed_dim=32, num_heads=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.tokenizer = FeatureTokenizer(num_features, embed_dim)
        self.layers = nn.ModuleList([
            SAINTBlock(embed_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.final_norm = nn.LayerNorm(embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        x = self.tokenizer(x)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        x = x.mean(dim=1)
        return self.classifier(x)

    def predict_proba(self, x):
        return torch.softmax(self.forward(x), dim=1)


def create_model(num_features=13, num_classes=3):
    return DLASS_PRP(
        num_features=num_features,
        num_classes=num_classes,
        embed_dim=32,
        num_heads=4,
        num_layers=3,
        dropout=0.1
    )