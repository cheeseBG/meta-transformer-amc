import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torchsummary import summary


class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.att_drop = nn.Dropout(0.1)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = nn.Dropout(0.1)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / (C // self.num_heads)**0.5
        attn = attn.softmax(dim=-1)
        attn = self.att_drop(attn)
        x = attn @ v
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class ViTBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_ratio):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, int(embed_dim * mlp_ratio), embed_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class ViT_v2(nn.Module):
    def __init__(self, in_channels, embed_dim, num_layers, num_heads, mlp_dim, num_classes, patch_size, in_size):
        super().__init__()
        self.patch_embed = PatchEmbedding(in_channels, embed_dim, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + (in_size[0]*in_size[1]) // (patch_size[0] * patch_size[1]), embed_dim))
        self.pos_drop = nn.Dropout(0.1)

        self.blocks = nn.Sequential(
            *[ViTBlock(embed_dim, num_heads, mlp_dim) for _ in range(num_layers)]
        )

        self.norm = nn.LayerNorm(embed_dim)

        self.clf_dense_1 = nn.Linear(embed_dim, 64)
        self.bn_1 = nn.BatchNorm1d(64)
        self.clf_drop_1 = nn.Dropout(p=0)
        
        self.clf_dense_2 = nn.Linear(64, 32)
        self.bn_2 = nn.BatchNorm1d(32)
        self.clf_drop_2 = nn.Dropout(p=0)

        self.clf_dense_3 = nn.Linear(32, 16)
        self.bn_3 = nn.BatchNorm1d(16)
        self.clf_drop_3 = nn.Dropout(p=0)
        
        self.clf_dense_4 = nn.Linear(16, num_classes)

    def forward(self, x):
        B, _, _, _ = x.shape
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.blocks(x)

        x = self.norm(x[:, 0])
    
        x = F.relu(self.clf_dense_1(x))
        x = self.clf_drop_1(self.bn_1(x))
        x = F.relu(self.clf_dense_2(x))
        x = self.clf_drop_2(self.bn_2(x))
        x = F.relu(self.clf_dense_3(x))
        x = self.clf_drop_3(self.bn_3(x))
        x = self.clf_dense_4(x)

        return x