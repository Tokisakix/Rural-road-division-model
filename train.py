import numpy as np
import torch
from torch import nn, optim, LongTensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import perf_counter
import os

from load_config import load_config
from data import get_dataset, Model
from dataloader import get_dataloader
from network import DeConvBn, ConvBn, Unet, Classifier
from network import DinkNet34, Dblock, DecoderBlock
from network import ViTEncoder, ViT
from logger import Logger

CONFIG        = load_config()
CUDA          = CONFIG["cuda"]
LOG_CONFIG    = CONFIG["log"]
LOG_ROOT      = LOG_CONFIG["root"]
SAVE_NUM      = LOG_CONFIG["save_num"]
logger        = Logger(LOG_ROOT, SAVE_NUM)

TRAIN_CONFIG  = CONFIG["train"]
LEARNING_RATE = TRAIN_CONFIG["learning_rate"]
EPOCHS        = TRAIN_CONFIG["epochs"]
SHOW_CONFIG   = CONFIG["show"]
SEG_LOSS_IMG  = os.path.join(logger.root, SHOW_CONFIG["seg_loss_img"])
CTA_LOSS_IMG  = os.path.join(logger.root, SHOW_CONFIG["cta_loss_img"])
CLASSIFIER_LOSS_IMG  = os.path.join(logger.root, SHOW_CONFIG["classifier_loss_img"])

def cal_Contra_loss(image, isroad):
    margin = 0.1
    loss   = 0.0
    n      = image.shape[0]

    a_idx = np.arange(n)
    b_idx = np.arange(n)

    np.random.shuffle(b_idx)
    for a, b in zip(a_idx, b_idx):
        a_image = image[a]
        b_image = image[b]
        if isroad[a] == isroad[b]:
            d   = torch.abs(a_image - b_image)
            loss+= d * d
        else:
            d   = torch.clamp(margin - torch.abs(a_image - b_image), min=0)
            loss+= d * d
    loss = loss.mean() / (2 * n)
    return loss

def check_road(labels, unusual_percent):
    res = []
    for label in labels:
        flag = (torch.sum(label) > 0) and (torch.sum(label) < 1024 * 1024  * 1 * unusual_percent)
        res.append(int(flag))
    res = torch.LongTensor(res)
    return res

def train(model, classifier, cam, seg_optimizer, seg_ceriterion, classifier_optimizer, classifier_ceriterion, dataloader, logger):
    start                = perf_counter()
    tot_seg_loss         = 0
    tot_cta_loss         = 0
    tot_classifier_loss  = 0
    epoch_list           = []
    seg_loss_list        = []
    cta_loss_list        = []
    classifier_loss_list = []

    for epoch in range(1, EPOCHS + 1):
        for (idx, inputs, labels, cleans, clean_label_path) in tqdm(dataloader):
            isroad   = check_road(labels, 0.2)
            inputs   = inputs.cuda() if CUDA else inputs
            labels   = labels.cuda() if CUDA else labels
            isroad   = isroad.cuda() if CUDA else isroad

            outputs  = model(inputs)

            seg_optimizer.zero_grad()
            seg_loss      = seg_ceriterion(outputs, labels)
            tot_seg_loss += seg_loss.cpu().item()
            cta_loss      = cal_Contra_loss(outputs, isroad)
            tot_cta_loss += cta_loss.cpu().item()
            seg_loss     += cta_loss
            seg_loss.backward()
            seg_optimizer.step()

            outputs              = classifier(outputs.detach())
            classifier_optimizer.zero_grad()
            classifier_loss      = classifier_ceriterion(outputs, isroad)
            classifier_loss.backward()
            tot_classifier_loss += classifier_loss.cpu().item()
            classifier_optimizer.step()

        seg_loss            = tot_seg_loss / len(dataloader)
        cta_loss            = tot_cta_loss / len(dataloader)
        classifier_loss     = tot_classifier_loss / len(dataloader)
        tot_seg_loss        = 0
        tot_classifier_loss = 0
        time                = perf_counter() - start
        start               = perf_counter()
        logger.info("\n------")
        logger.info(f"Epoch:{epoch:3d} Seg Loss:{seg_loss:10.6f} Cta Loss:{cta_loss:10.6f} Classifier Loss:{classifier_loss:10.6f} Time:{time:6.2f}s.")
        logger.save_model(model, classifier, f"Epoch_{epoch}_Seg.pth", f"Epoch_{epoch}_Classifier.pth")
        logger.info(f"Save model as Epoch_{epoch}_Seg.pth | Epoch_{epoch}_Classifier.pth")
        epoch_list.append(epoch)
        seg_loss_list.append(seg_loss)
        cta_loss_list.append(cta_loss)
        classifier_loss_list.append(classifier_loss)

    # for (raw_idx, raw_inputs, raw_label, raws, raw_label_path) in raw_dataloader:
    #     raw_inputs  = raw_inputs.cuda() if CUDA else raw_inputs
    #     raw_outputs = model(raw_inputs).detach()
    #     raw_predict = classifier(raw_outputs).detach()
    #     for idx, index in enumerate(raw_idx):
    #         (image, label, clean, label_path) = raw_dataloader.dataset.dataset[index]
    #         score = raw_predict[idx]
    #         mask  = raw_outputs[idx]
    #         if score > clean.item():
    #             # raw_dataloader.dataset.update(index, image, mask, score, label_path)
    #             pass

    logger.info("Finished training!")
    return  epoch_list, seg_loss_list, cta_loss_list, classifier_loss_list

def draw(epoch_list, seg_loss_list, cta_loss_list, seg_classifier_list):
    plt.plot(epoch_list, seg_loss_list, label="Seg Loss")
    plt.title("Loss Img")
    plt.legend()
    plt.savefig(SEG_LOSS_IMG)
    plt.close()

    plt.plot(epoch_list, seg_classifier_list, label="Classifier Loss")
    plt.title("Loss Img")
    plt.legend()
    plt.savefig(CLASSIFIER_LOSS_IMG)
    plt.close()

    plt.plot(epoch_list, cta_loss_list, label="Cta Loss")
    plt.title("Loss Img")
    plt.legend()
    plt.savefig(CTA_LOSS_IMG)
    plt.close()
    return

if __name__ == "__main__":
    logger.info("Logger init.")

    clean_dataset = get_dataset(CONFIG, clean=True)
    # raw_dataset   = get_dataset(CONFIG, clean=False)

    clean_dataloader = get_dataloader(CONFIG, clean_dataset, clean=True)
    # raw_dataloader   = get_dataloader(CONFIG, raw_dataset, clean=True)
    logger.info("Load data.")

    model      = Unet()
    classifier = Classifier()
    model      = model.cuda() if CUDA else model
    classifier = classifier.cuda() if CUDA else classifier
    logger.info("Build model.")

    seg_optimizer         = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    classifier_optimizer  = optim.Adam(classifier.parameters(), lr=LEARNING_RATE)
    seg_ceriterion        = nn.BCEWithLogitsLoss()
    classifier_ceriterion = nn.CrossEntropyLoss()
    # FIXME.CAM的使用
    # cam 目前已作为参数传入到 train() 函数中，具体用法依据之前讨论还未定下，故目前 CAM 在 train() 函数中是零作用
    cam                  = None

    epoch_list, loss_list, cta_loss_list, seg_classifier_list = train(model, classifier, cam, seg_optimizer, seg_ceriterion,
                                                      classifier_optimizer, classifier_ceriterion, clean_dataloader,
                                                      logger)
    draw(epoch_list, loss_list, cta_loss_list, seg_classifier_list)
    # clean_dataset.save()
    # raw_dataset.save()
