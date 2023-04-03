import torch
import torch.nn as nn


# not used
# class EmbedBlock(nn.Module):
#     """ Restrictive CNN Block """
#     def __init__(self, dim, in_channels, out_channels):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(dim, eps=1e-6)
#         self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
#         self.norm2 = nn.LayerNorm(dim, eps=1e-6)
#         self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=1, stride=1, padding=0)
#         self.conv3 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2, padding=0)
#
#     def forward(self, x):
#         x_ = self.conv1(self.norm1(x))
#         x_ = self.conv2(self.norm2(x_))
#         x = self.conv3(x + x_)
#
#         return x


class TokenEmbedding(nn.Module):
    """ Tokenize the image """
    def __init__(self, img_size=256, patch_size=16, in_channels=3):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.embed_dim = patch_size * patch_size * in_channels

        # Patch embedding
        self.proj = nn.Conv2d(in_channels=in_channels, out_channels=self.embed_dim, kernel_size=patch_size, stride=patch_size)

        # Position embedding
        self.position = nn.Parameter(torch.ones(self.num_patches, self.embed_dim))

    def forward(self, x):
        x = self.proj(x)

        x = x.flatten(2)
        x = x.transpose(1, 2)

        x += self.position
        return x


class MultiHeadAttention(nn.Module):
    """Attention mechanism.
       Parameters
       ----------
       dim : int
           The input and out dimension of per token features.
       n_heads : int
           Number of attention heads.
       qkv_bias : bool
           If True then we include bias to the query, key and value projections.
       att_p : float
           Dropout probability applied to the query, key and value tensors.
       proj_p : float
           Dropout probability applied to the output tensor.
       Attributes
       ----------
       scale : float
           Normalizing consant for the dot product.
       qkv : nn.Linear
           Linear projection for the query, key and value.
       proj : nn.Linear
           Linear mapping that takes in the concatenated output of all attention
           heads and maps it into a new space.
       attn_drop, proj_drop : nn.Dropout
           Dropout layers.
       """

    def __init__(self, dim, n_heads=12, qkv_bias=True, att_p=0., proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(att_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        n_samples, n_tokens, dim = x.shape

        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x)  # (n_samples, n_patches + 1, 3 * dim)
        qkv = qkv.reshape(
            n_samples, n_tokens, 3, self.n_heads, self.head_dim
        )  # (n_smaples, n_patches + 1, 3, n_heads, head_dim)
        qkv = qkv.permute(
            2, 0, 3, 1, 4
        )  # (3, n_samples, n_heads, n_patches + 1, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1)  # (n_samples, n_heads, head_dim, n_patches + 1)
        dp = (
                     q @ k_t
             ) * self.scale  # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = dp.softmax(dim=-1)  # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v  # (n_samples, n_heads, n_patches +1, head_dim)
        weighted_avg = weighted_avg.transpose(
            1, 2
        )  # (n_samples, n_patches + 1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2)  # (n_samples, n_patches + 1, dim)

        x = self.proj(weighted_avg)  # (n_samples, n_patches + 1, dim)
        x = self.proj_drop(x)  # (n_samples, n_patches + 1, dim)

        return x


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = nn.functional.gelu
        self.drop = nn.Dropout(p)

    def forward(self, x):
        x = self.fc1(x)
        x = self.drop(self.act(x))
        x = self.fc2(x)
        x = self.drop(self.act(x))

        return x


class Block(nn.Module):
    """ Transformer Encoder Block """
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., att_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.att = MultiHeadAttention(dim, n_heads=n_heads, qkv_bias=qkv_bias, att_p=att_p, proj_p=p)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=hidden_features, out_features=dim)

    def forward(self, x):
        x = x + self.att(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x


class ColorTransformer(nn.Module):
    def __init__(self, img_size=256, in_channels=3, out_channels=2, patch_size=16, embed_dim=768, encoder_depth=12,
                 n_heads=8, mlp_ratio=4., qkv_bias=True, p=0., att_p=0.):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.embed = TokenEmbedding(img_size=img_size, patch_size=patch_size, in_channels=in_channels)

        self.blocks = nn.ModuleList([
            Block(dim=self.embed.embed_dim, n_heads=n_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, p=p, att_p=att_p)
            for _ in range(encoder_depth)
        ])

        self.linear_proj = nn.Conv2d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1, stride=1)

    def forward(self, x, hint):
        x = torch.cat([x, hint], dim=1)

        x = self.embed(x)
        for block in self.blocks:
            x = block(x)

        # Unflatten
        x = x.transpose(1, 2)
        # x = x.reshape([-1, x.shape[1], int(x.shape[2] ** 0.5), int(x.shape[2] ** 0.5)])
        x = x.reshape([-1, self.in_channels, self.img_size, self.img_size])

        # x = self.decoder(x)
        x = self.linear_proj(x)

        return x


if __name__ == '__main__':
    input = torch.zeros([1, 3, 256, 256])
    t = ColorTransformer()
    print(t.forward(input).shape)
