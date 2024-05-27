import torch
# import gradio as gr
import cv2 as cv
import numpy as np
from load_config import load_config

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
    image           = image.permute(2, 0, 1).unsqueeze(0)
    feature, logits = MODEL(image)
    logits          = logits.detach().squeeze(0).cpu().permute(1, 2, 0).numpy()  # [H, W, 1]
    logits_image    = np.uint8(logits* 255)
    logits_image    = cv.cvtColor(logits_image, cv.COLOR_RGB2BGRA)
    logits_image[logits_image<128]  = 0
    logits_image[logits_image>=128] = 255
    # 可以考虑加入阈值处理增加图像对比度
    # _, thresholded_image = cv.threshold(logits_image, 128, 255, cv.THRESH_BINARY)

    return logits_image  # 返回处理后的图像


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
