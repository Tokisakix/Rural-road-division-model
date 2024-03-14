import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        return

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
