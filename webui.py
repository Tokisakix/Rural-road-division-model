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
SHARE      = WEBUI["share"]
PORT       = WEBUI["port"]
MODEL_PATH = WEBUI["model_path"]
MODEL      = load_model(MODEL_PATH)

def predict(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB) / 255
    image = (torch.FloatTensor(image).cuda() if CUDA else torch.FloatTensor(image)).permute(2, 0, 1)
    image = image.unsqueeze(0)
    out = MODEL(image).detach()
    out = out.squeeze(0)
    out = out.cpu().permute(1, 2, 0)
    out = cv.cvtColor(np.uint8(out * 255), cv.COLOR_RGB2BGRA)
    # out[out > 0.5] = 1
    # out[out <= 0.5] = 0
    return out

# image = cv.imread("/public/zjj/public/zjj/jy/work5/data/croproad_dataset/test/tvbz21m600c2_1 (2).png")
# predict(image)
# print("ok")
interface = gr.Interface(fn=predict, inputs="image", outputs="image")

interface.launch(share=SHARE, server_port=PORT)

