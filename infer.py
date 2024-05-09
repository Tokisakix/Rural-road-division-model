import torch
# import gradio as gr
import numpy as np
import cv2 as cv
import argparse
from load_config import load_config
from network import UNet

CONFIG     = load_config()
CUDA       = CONFIG["cuda"]
MODEL_PATH = CONFIG["infer"]["model_path"]
MODEL      = torch.load(MODEL_PATH)
MODEL      = MODEL.cuda() if CUDA and torch.cuda.is_available() else MODEL

if isinstance(MODEL, torch.nn.DataParallel):
    MODEL = MODEL.module

MODEL.eval()

def predict(image):
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB) / 255
    image = torch.FloatTensor(image)
    if CUDA and torch.cuda.is_available():
        image = image.cuda()
    image = image.permute(2, 0, 1).unsqueeze(0)
    out   = MODEL(image).detach().squeeze(0).cpu().permute(1, 2, 0)
    out   = cv.cvtColor(np.uint8(out * 255), cv.COLOR_RGB2BGRA)
    out[out >  128] = 255
    out[out <= 128] = 0
    return out

use_ui      = False
input_path  = '/public/zjj/public/zjj/xzx/data-road-clipped/clean/image/1000.jpg'
output_path = 'infer.jpg'

if use_ui:
    pass
    # # UI
    # INFER = CONFIG["infer"]
    # SHARE = INFER["share"]
    # PORT  = INFER["port"]
    # interface = gr.Interface(fn=predict, inputs="image", outputs="image")
    # interface.launch(share=SHARE, server_port=PORT)
else:
    input_image  = cv.imread(input_path)
    output_image = predict(input_image)
    cv.imwrite(output_path, output_image)
