import torch.optim as optim
import os
import shutil

from framework import FrameWork
from utils import load_config
from utils.loader import get_loader
from utils.logger import LogPrinter
from utils.model import Model, load_model, save_model
from utils.criterion import DicBceLoss

# config 设置
CONFIG       = load_config()
CUDA         = CONFIG["cuda"]
TRAIN_CONFIG = CONFIG["train"]
LOG_CONFIG   = CONFIG["log"]
PRO_NAME     = TRAIN_CONFIG["name"]
PRETRAINED   = TRAIN_CONFIG["pretrained"]
EPOCHS       = TRAIN_CONFIG["train_epochs"]
LR           = TRAIN_CONFIG["learning_rate"]
SHOW_STEP    = TRAIN_CONFIG["show_info_step"]
TEST_STEP    = TRAIN_CONFIG["test_info_step"]
SAVE_STEP    = TRAIN_CONFIG["save_model_step"]
SAVE_NUM     = TRAIN_CONFIG["save_model_num"]
LOG_ROOT     = LOG_CONFIG["root"] + PRO_NAME

# 初始化日志
shutil.rmtree(LOG_ROOT) if os.path.isdir(LOG_ROOT) else None
os.mkdir(LOG_ROOT)
logger = LogPrinter(LOG_ROOT, SAVE_NUM)
logger.log(f"[INFO] Log's root is {LOG_ROOT}.")

# 获取模型
model = load_model("pretrained/pretrained.pth") if CUDA else Model()
model = model.cuda() if CUDA else model
logger.log("[INFO] " + ("Use" if PRETRAINED else "Not") + " pretrained model.")
logger.log("[INFO] Model train in " + ("CUDA." if CUDA else "CPU."))

# 数据加载器
train_loader, test_loader = get_loader()
logger.log("[INFO] Loaded data loader.")

# 优化器
optimizer = optim.Adam(params=model.parameters(), lr=LR)
logger.log(f"[INFO] Use {optimizer.__class__.__name__} optimizer.")

# 损失函数
criterion = DicBceLoss()
logger.log(f"[INFO] Use {criterion.__class__.__name__} criterion.")

# 定义框架
framework = FrameWork(
    model = model,
    logger = logger,
    train_loader = train_loader,
    test_loader = test_loader,
)

# 训练
framework.train(
    CUDA = CUDA, 
    EPOCHS = EPOCHS, 
    SHOW_STEP = SHOW_STEP, 
    TEST_STEP = TEST_STEP, 
    SAVE_STEP = SAVE_STEP,
)