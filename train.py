import torch
from torch import nn, optim
import matplotlib.pyplot as plt
from tqdm import tqdm
from time import perf_counter
import os

from load_config import load_config
from data import get_dataset, Model
from dataloader import get_dataloader
from model import Unet, Classifer
from model import DinkNet34, Dblock, DeConvBn, DecoderBlock
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
CLASSIFER_LOSS_IMG  = os.path.join(logger.root, SHOW_CONFIG["classifer_loss_img"])

def train(model, classifer, seg_optimizer, seg_ceriterion, classifer_optimizer, classifer_ceriterion, clean_dataloader, raw_dataloader, logger):
    start = perf_counter()
    tot_seg_loss = 0
    tot_classifer_loss = 0
    epoch_list = []
    seg_loss_list = []
    classifer_loss_list = []

    for epoch in range(1, EPOCHS + 1):
        for (clean_idx, clean_inputs, clean_labels, cleans, clean_label_path), (raw_idx, raw_inputs, raw_label, raws, raw_label_path) in tqdm(zip(clean_dataloader, raw_dataloader)):
            clean_inputs = clean_inputs.cuda() if CUDA else clean_inputs
            clean_labels = clean_labels.cuda() if CUDA else clean_labels
            raw_inputs   = raw_inputs.cuda() if CUDA else raw_inputs
            cleans       = cleans.cuda() if CUDA else cleans
            raws         = raws.cuda() if CUDA else raws

            clean_outputs = model(clean_inputs)
            raw_outputs   = model(raw_inputs)
            seg_optimizer.zero_grad()
            seg_loss = seg_ceriterion(clean_outputs, clean_labels)
            seg_loss.backward()
            tot_seg_loss += seg_loss.cpu().item()
            seg_optimizer.step()

            clean_outputs = clean_labels.detach()
            raw_outputs   = raw_outputs.detach()
            clean_score   = classifer(clean_outputs)
            raw_score     = classifer(raw_outputs)
            classifer_optimizer.zero_grad()
            classifer_loss= classifer_ceriterion(clean_score, cleans) + classifer_ceriterion(raw_score, raws)
            classifer_loss.backward()
            tot_classifer_loss += classifer_loss.cpu().item()
            classifer_optimizer.step()

        seg_loss = tot_seg_loss / len(clean_dataloader)
        classifer_loss = tot_classifer_loss / len(clean_dataloader)
        tot_seg_loss = 0
        tot_classifer_loss = 0
        time  = perf_counter() - start
        start = perf_counter()
        logger.info("\n------")
        logger.info(f"Epoch:{epoch:3d} Seg Loss:{seg_loss:10.6f} Classifer Loss:{classifer_loss:10.6f} Time:{time:6.2f}s.")
        logger.save_model(model, classifer, f"Epoch_{epoch}_Seg.pth", f"Epoch_{epoch}_Classifer.pth")
        logger.info(f"Save model as Epoch_{epoch}_Seg.pth | Epoch_{epoch}_Classifer.pth")
        epoch_list.append(epoch)
        seg_loss_list.append(seg_loss)
        classifer_loss_list.append(classifer_loss)

    for (raw_idx, raw_inputs, raw_label, raws, raw_label_path) in raw_dataloader:
        raw_inputs  = raw_inputs.cuda() if CUDA else raw_inputs
        raw_outputs = model(raw_inputs).detach()
        raw_predict = classifer(raw_outputs).detach()
        for idx, index in enumerate(raw_idx):
            (image, label, clean, label_path) = raw_dataloader.dataset.dataset[index]
            score = raw_predict[idx]
            mask  = raw_outputs[idx]
            if score > clean.item():
                raw_dataloader.dataset.update(index, image, mask, score, label_path)

    logger.info("Finished training!")
    return  epoch_list, seg_loss_list, classifer_loss_list

def draw(epoch_list, seg_loss_list, seg_classifer_list):
    plt.plot(epoch_list, seg_loss_list, label="Seg Loss")
    plt.title("Loss Img")
    plt.legend()
    plt.savefig(SEG_LOSS_IMG)
    plt.close()

    plt.plot(epoch_list, seg_classifer_list, label="Classifer Loss")
    plt.title("Loss Img")
    plt.legend()
    plt.savefig(CLASSIFER_LOSS_IMG)
    plt.close()
    return




if __name__ == "__main__":
    logger.info("Logger init.")

    clean_dataset = get_dataset(CONFIG, clean=True)
    raw_dataset   = get_dataset(CONFIG, clean=False)

    clean_dataloader = get_dataloader(CONFIG, clean_dataset, clean=True)
    raw_dataloader   = get_dataloader(CONFIG, raw_dataset, clean=True)
    logger.info("Load data.")

    model     = Unet()
    classifer = Classifer()
    model     = model.cuda() if CUDA else model
    classifer = classifer.cuda() if CUDA else classifer
    logger.info("Build model.")

    seg_optimizer        = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    classifer_optimizer  = optim.Adam(classifer.parameters(), lr=LEARNING_RATE)
    seg_ceriterion       = nn.BCEWithLogitsLoss()
    classifer_ceriterion = nn.MSELoss()

    epoch_list, loss_list, seg_classifer_list = train(model, classifer, seg_optimizer, seg_ceriterion, classifer_optimizer, classifer_ceriterion, clean_dataloader, raw_dataloader, logger)
    draw(epoch_list, loss_list, seg_classifer_list)
    clean_dataset.save()
    raw_dataset.save()