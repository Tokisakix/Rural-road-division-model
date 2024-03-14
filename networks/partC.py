import torch
from torch import nn, optim
import json
import torch.nn.functional as F

from utils import load_config
from utils.model import load_model
from .vit import ViT

TEST_BATCH_SIZE = 2
CONFIG_PATH = "config.json"
CONFIG     = load_config()
CUDA       = CONFIG["cuda"]
WEBUI      = CONFIG["webui"]
TEST       = CONFIG["test"]
SHARE      = WEBUI["share"]
PORT       = WEBUI["port"]
source_root= TEST["source_root"]
target_file= TEST["target_file"]

def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)
    return config

class PatchEmbedding(nn.Module):
    def __init__(self, patch_size, in_channels, embed_dim):
        super(PatchEmbedding, self).__init__()
        self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        out = self.conv(x)
        return out
    
class InfoNCE(nn.Module):
    def __init__(self, temperature=1.0):
        super(InfoNCE, self).__init__()
        self.temperature = temperature
        self.criterion = nn.BCEWithLogitsLoss()  # 考虑nn.CrossEntropyLoss()

    def forward(self, imgs, positive, negative):   # imgs是自身编码、positive也用自身编码,negative是同batch其他图片编码
        imgs = F.normalize(imgs, dim=-1, p=2)
        positive = F.normalize(positive, dim=-1, p=2)
        negative = F.normalize(negative, dim=-1, p=2)

        pos_similarity = torch.sum(imgs * positive, dim=-1) / self.temperature
        neg_similarity = torch.sum(imgs * negative, dim=-1) / self.temperature

        similarities = torch.cat([pos_similarity, neg_similarity], dim=0)

        labels = torch.zeros_like(similarities)
        pos_position = pos_similarity.size(0)
        labels[:pos_position] = 1

        loss = self.criterion(similarities, labels.float())

        return loss
    
class PartC(nn.Module):
    def __init__(self, vit_model, classify_model, optimizer, contrast_loss, classify_loss, vit_learning_rate, classify_learning_rate):
        """
        vit_model     -> ViTEncoder(
            img_size,
            patch_size,
            in_channels,
            embed_dim,
            num_heads,
            num_layers,
            mlp_hidden_dim,
            use_cls,
        )
        optimizer     -> torch.optim.Adam
        contrast_loss -> nn.CrossEntropyLoss()
        """
        super(PartC, self).__init__()
        self.vit = vit_model
        self.classify = classify_model
        self.optim_vit = optimizer(self.vit.parameters(), lr = vit_learning_rate)
        self.optim_classify = optimizer(self.classify.parameters(), lr = classify_learning_rate)
        self.optim_contrast = optimizer(self.vit.parameters(), lr=0.001)
        self.contrast_loss = contrast_loss
        self.classify_loss = classify_loss
        return
    
    def train_contrast_loss(self, outputs, imgs, labels, is_positive): #(self,anchor, positive, negative)
        """
        imgs        -> FloatTensor[batch * 16, 3, 256, 256]
        labels      -> FloatTensor[batch * 16, 256, 256]
        outputs     -> FloatTensor[batch * 16, 256, 256]
        is_positive -> FloatTensor[batch * 16]
        """
        imgs = self.vit(imgs)
        loss = self.contrast_loss(imgs, positive, negative)

        self.optim_contrast.zero_grad()
        loss.backward()
        self.optim_contrast.step()
        
        return loss.item() #1.8
    
    def train_classify_loss(self, outputs, imgs, labels, is_positive):
        """
        imgs        -> FloatTensor[batch * 16, 3, 256, 256]
        labels      -> FloatTensor[batch * 16, 256, 256]
        outputs     -> FloatTensor[batch * 16, 256, 256]
        is_positive -> FloatTensor[batch * 16]
        """
        classes = self.classify(outputs)
        loss = self.classify_loss(classes, is_positive)

        self.optim_classify.zero_grad()
        loss.backward()
        print("classify loss:", loss)
        self.optim_classify.step()
        return loss.item()

    def train(self, imgs, labels):
        """
        imgs        -> FloatTensor[batch * 16, 3, 256, 256]
        labels      -> FloatTensor[batch * 16, 256, 256]
        outputs     -> FloatTensor[batch * 16, 256, 256]
        is_positive -> FloatTensor[batch * 16]
        """
        if (torch.sum(labels) == 0):
           is_positive = torch.zeros(imgs.shape[0]).long().cuda()
           print("negative")
           outputs = self.vit(imgs)
           contrast_loss = self.train_contrast_loss(outputs, imgs, labels, is_positive)
           loss = contrast_loss
        else:
           is_positive = torch.ones(imgs.shape[0]).long().cuda()
           print("positive")
           outputs = self.vit(imgs)
           contrast_loss = self.train_contrast_loss(outputs, imgs, labels, is_positive)
           classify_loss = self.train_classify_loss(outputs, imgs, labels, is_positive)
           loss = contrast_loss + classify_loss

        return loss, outputs

    
    def forward(self, imgs):
        outputs = self.vit(imgs)
        return outputs
    
def get_PartC():

    PART_C_CONFIG = load_config()["model"]["PartC"]
    vit_model = ViT(
        img_size = PART_C_CONFIG["img_size"],
        patch_size = PART_C_CONFIG["patch_size"],
        in_channels = PART_C_CONFIG["in_channels"],
        embed_dim = PART_C_CONFIG["embed_dim"],
        num_heads = PART_C_CONFIG["num_heads"],
        num_layers = PART_C_CONFIG["num_layers"],
        mlp_hidden_dim = PART_C_CONFIG["mlp_hidden_dim"],
        num_classes = PART_C_CONFIG["output_token_size"],
        use_cls = False,
    )
    classify_model = nn.Sequential(
        nn.Linear(PART_C_CONFIG["classify_dim"], 2),
        nn.Sigmoid(),
    )
    optimizer = optim.Adam
    contrast_loss = InfoNCE()
    classify_loss = nn.CrossEntropyLoss()

    part_c = PartC(
        vit_model,
        classify_model,
        optimizer,
        contrast_loss,
        classify_loss,
        vit_learning_rate = PART_C_CONFIG["vit_learning_rate"],
        classify_learning_rate = PART_C_CONFIG["classify_learning_rate"],
    )
    return part_c

if __name__ == "__main__":
    imgs        = torch.rand(TEST_BATCH_SIZE * 16, 3, 256, 256)
    labels      = torch.rand(TEST_BATCH_SIZE * 16, 256, 256)
    is_positive = torch.ones(TEST_BATCH_SIZE * 16).long()
    
    partC = get_PartC()
    outputs = partC(imgs)
    partC.train(imgs, labels, is_positive)
    print("tokens.shape", outputs.shape)