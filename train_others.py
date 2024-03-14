import torch
import torch.optim as optim
import os
import shutil

from networks.dinknet import DinkNet34

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
from framework import FrameWork, OtherPredict
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

# 数据加载器
train_loader, test_loader = get_loader()
logger.log("[INFO] Loaded data loader.")

# 定义框架
framework = OtherPredict(
    model = DinkNet34(),
    criterion = torch.nn.MSELoss(),
    optimizer = torch.optim.Adam,
    logger = logger,
    train_loader = train_loader,
    test_loader = test_loader,
    CUDA = CUDA,
    Devices=[0],
)

# 训练
framework.train(
    CUDA = CUDA, 
    EPOCHS = EPOCHS, 
    SHOW_STEP = SHOW_STEP, 
    TEST_STEP = TEST_STEP, 
    SAVE_STEP = SAVE_STEP,
)