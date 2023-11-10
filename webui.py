import torch
import gradio as gr
import numpy as np
import cv2 as cv

from utils import load_config
from utils.model import load_model

CONFIG     = load_config()
CUDA       = CONFIG["cuda"]
WEBUI      = CONFIG["webui"]
SHARE      = WEBUI["share"]
PORT       = WEBUI["port"]
MODEL_PATH = WEBUI["model_path"]
MODEL      = load_model(MODEL_PATH)

def predict(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB) / 255
    image = (torch.tensor(image).cuda() if CUDA else torch.tensor(image)).permute(2, 0, 1)
    out = torch.rand(1, 1024, 1024).cuda()
    out = out.cpu().permute(1, 2, 0)
    out = cv.cvtColor(np.uint8(out * 255), cv.COLOR_RGB2BGRA)
    return out

interface = gr.Interface(fn=predict, inputs="image", outputs="image")

interface.launch(share=SHARE, server_port=PORT)