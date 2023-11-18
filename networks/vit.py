import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        out = self.conv(x)
        return out
    
class ResNet(nn.Module):
    def __init__(self, layer):
        super(ResNet, self).__init__()
        self.layer = layer
        return
    
    def forward(self, x):
        out = x + self.layer(x)
        return out
    
class MultiSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiSelfAttention, self).__init__()
        self.func = nn.MultiheadAttention(embed_dim, num_heads)
        return
    
    def forward(self, x):
        out, _ = self.func(x, x, x)
        return out

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_hidden_dim, dropout=0.5):
        super(TransformerEncoderBlock, self).__init__()
        self.attention = ResNet(nn.Sequential(
            MultiSelfAttention(embed_dim, num_heads),
        ))
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = ResNet(nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden_dim),
            nn.GELU(),
            nn.Linear(ffn_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        ))
        self.norm2 = nn.LayerNorm(embed_dim)
        return

    def forward(self, x):
        out = self.attention(x)
        out = self.norm1(out)
        out = self.ffn(out)
        out = self.norm2(out)
        return out
    
class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_hidden_dim, num_layers, dropout=0.5):
        super(TransformerEncoderLayer, self).__init__()
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim, num_heads, ffn_hidden_dim, dropout) for _ in range(num_layers)
        ])

    def forward(self, x):
        out = x
        for block in self.transformer_blocks:
            out = block(out)
        return out

class ViTEncoder(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, num_heads, num_layers, mlp_hidden_dim, use_cls=True, dropout=0.5):
        super(ViTEncoder, self).__init__()
        self.patch_embedding = PatchEmbedding(patch_size, in_channels, embed_dim)
        self.transformer = TransformerEncoderLayer(embed_dim, num_heads, mlp_hidden_dim, num_layers, dropout)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim)) if use_cls else None
        self.positional_embedding = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2 + (1 if use_cls else 0), embed_dim))
        return

    def forward(self, x):
        x = self.patch_embedding(x)
        x = x.flatten(2).transpose(1, 2)
        if self.cls_token != None:
            cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.positional_embedding
        x = self.transformer(x)
        return x

if __name__ == "__main__":
    img_size = 256
    patch_size = 16
    in_channels = 3
    embed_dim = 768
    num_heads = 4
    num_layers = 4
    mlp_hidden_dim = 1024
    use_cls = False

    model   = ViTEncoder(img_size, patch_size, in_channels, embed_dim, num_heads, num_layers, mlp_hidden_dim, use_cls)
    inputs  = torch.rand(4, in_channels, img_size, img_size)
    outputs = model(inputs)
    print(model)
    print("inputs's shape : ", inputs.shape)
    print("outputs's shaoe: ", outputs.shape)
