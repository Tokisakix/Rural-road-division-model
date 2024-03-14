import os
import time
import torch
import gradio as gr
import numpy as np
import cv2 as cv
from torch.nn import Linear

from utils import load_config
from utils.model import load_model
from networks.dinknet import Dblock, DecoderBlock, DinkNet34, Model

CONFIG     = load_config()
CUDA       = CONFIG["cuda"]
WEBUI      = CONFIG["webui"]
TEST       = CONFIG["test"]
SHARE      = WEBUI["share"]
PORT       = WEBUI["port"]
MODEL_PATH = WEBUI["model_path"]
MODEL     = load_model(MODEL_PATH)
source_root= TEST["source_root"]
target_file= TEST["target_file"]

def predict(image):
    image = cv.cvtColor(image,cv.COLOR_BGR2RGB) / 255
    image = (torch.FloatTensor(image).cuda() if CUDA else torch.FloatTensor(image)).permute(2, 0, 1)
    image = image.unsqueeze(0)
    out = MODEL(image).detach()
    out = out.squeeze(0)
    out = out.cpu().permute(1, 2, 0)
    out = cv.cvtColor(np.uint8(out * 255), cv.COLOR_RGB2BGRA)
    # out[out > 0.5] = 1
    # out[out <= 0.5] = 0
    return out

val = os.listdir(source_root)
tic = time.time()
os.mkdir(target_file)

for i, name in enumerate(val):
     #if i % 10 == 0:
     #    print(i / 10, '    ', '%.2f' % (time() - tic))
     image = cv.imread(source_root + name)
     mask = predict(image)
     cv.imwrite(target_file + name[:-7] + 'mask.png', mask.astype(np.uint8))