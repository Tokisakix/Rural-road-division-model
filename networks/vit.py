import torch
import torch.nn as nn

from .transformer import PatchEmbedding, TransformerEncoderLayer

class ViTEncoder(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, num_heads, num_layers, mlp_hidden_dim, use_cls=True, dropout=0.5):
        super(ViTEncoder, self).__init__()
        self.patch_embedding = PatchEmbedding(patch_size, in_channels, embed_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim)) if use_cls else None
        self.positional_embedding = nn.Parameter(torch.randn(1, (img_size // patch_size) ** 2 + (1 if use_cls else 0), embed_dim))
        self.transformer = TransformerEncoderLayer(embed_dim, num_heads, mlp_hidden_dim, num_layers, dropout)
        self.norm = nn.LayerNorm(embed_dim)
        return

    def forward(self, x):
        out = self.patch_embedding(x)
        out = out.flatten(2).transpose(1, 2)
        if self.cls_token != None:
            cls_tokens = self.cls_token.expand(out.shape[0], -1, -1)
            out = torch.cat((cls_tokens, out), dim=1)
        out = out + self.positional_embedding
        out = self.transformer(out)
        out = self.norm(out)
        return out
    
class ViT(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim, num_heads, num_layers, mlp_hidden_dim, num_classes, use_cls=True, dropout=0.5):
        super(ViT, self).__init__()
        self.encoder = ViTEncoder(img_size, patch_size, in_channels, embed_dim, num_heads, num_layers, mlp_hidden_dim, use_cls, dropout)
        self.linear = nn.Sequential(
            nn.Flatten(),
            nn.Linear(embed_dim, num_classes),
            nn.Softmax(dim=-1),
        )
        return
    
    def forward(self, out):
        out = self.encoder(out)
        out = out.mean(dim=1)
        out = self.linear(out)
        return out


if __name__ == "__main__":
    img_size = 224
    patch_size = 16
    in_channels = 3
    embed_dim = 768
    num_heads = 4
    num_layers = 4
    mlp_hidden_dim = 1024
    num_classes = 1000  # 2
    use_cls = False

    # model   = ViTEncoder(img_size, patch_size, in_channels, embed_dim, num_heads, num_layers, mlp_hidden_dim, use_cls)
    model   = ViT(img_size, patch_size, in_channels, embed_dim, num_heads, num_layers, mlp_hidden_dim, num_classes)
    inputs  = torch.rand(4, in_channels, img_size, img_size)
    outputs = model(inputs)
    print(model)
    print("inputs's shape : ", inputs.shape)
    print("outputs's shape: ", outputs.shape)
    print("output:",outputs)
