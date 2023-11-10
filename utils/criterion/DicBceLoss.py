import torch
import torch.nn as nn

class DicBceLoss(nn.Module):
    """
    input: output_img - FloatTensor[batch_size, height, weight]
           label_img  - FloatTensor[batch_size, height, weight]
    output: DicBceLoss - FloatTensor
    """
    def __init__(self):
        super(DicBceLoss, self).__init__()
        self.bce_loss = nn.BCELoss()
        return
        
    def soft_dice_coefficient(self, label, output):
        smooth = 1e-5
        i = torch.sum(label)
        j = torch.sum(output)
        intersection = torch.sum(label * output)
        score = (2 * intersection + smooth) / (i + j + smooth)
        score = score.mean()
        return score

    def soft_dice_loss(self, label, output):
        loss = 1 - self.soft_dice_coefficient(label, output)
        return loss
        
    def __call__(self, output, label):
        loss1 = self.bce_loss(output, label)
        loss2 = self.soft_dice_loss(label, output)
        loss = loss1 + loss2
        return loss