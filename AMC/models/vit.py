import torch
import torch.nn as nn
import math
from torchsummary import summary

class PatchEmbedding(nn.Module):
    def __init__(self, in_channels, patch_size, embed_dim):
        super().__init__()
        self.patch_size = patch_size
        self.projection = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2),
        )

    def forward(self, x):
        x = self.projection(x)
        return torch.transpose(x, 1, 2)


class ViT(nn.Module):
    def __init__(self, in_channels, embed_dim, num_layers, num_heads, mlp_dim, patch_size, num_classes=None):
        super(ViT, self).__init__()
        self.patch_embed = PatchEmbedding(in_channels, patch_size, embed_dim)

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + (4 * 1024) // (patch_size*patch_size), embed_dim))

        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(nn.TransformerEncoderLayer(embed_dim, num_heads, mlp_dim))

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        for layer in self.layers:
            x = layer(x)

        x = self.norm(x)

        # flatten
        x = x.view(x.size(0), -1)
        return x


if __name__ == "__main__":
    in_channels = 1
    embed_dim = 768
    num_layers = 12
    num_heads = 12
    mlp_dim = 3072

    model = ViT(in_channels, embed_dim, num_layers, num_heads, mlp_dim)
    input_tensor = torch.randn(11, in_channels, 256, 256)

    output = model(input_tensor)
    print(output.shape)  # (batch_size, 1 + (height * width) // (patch_size * patch_size), embed_dim)