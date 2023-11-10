import torch.optim as optim
import os

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
EPOCH        = TRAIN_CONFIG["train_epoch"]
LR           = TRAIN_CONFIG["learning_rate"]
SHOW_STEP    = TRAIN_CONFIG["show_info_step"]
TEST_STEP    = TRAIN_CONFIG["test_info_step"]
LOG_ROOT     = LOG_CONFIG["root"] + PRO_NAME
LOG_PATH     = LOG_ROOT + "/log.txt"

# 初始化日志
os.mkdir(LOG_ROOT) if not os.path.isdir(LOG_ROOT) else None
logger = LogPrinter(LOG_PATH)

# 获取模型
model = load_model("pretrained/pretrained.pth") if CUDA else Model()
model = model.cuda() if CUDA else model

# 数据加载器
train_loader, test_loader = get_loader()

# 优化器
optimizer = optim.Adam(params=model.parameters(), lr=LR)

# 损失函数
criterion = DicBceLoss()

# save_model(model, "models/model.pth")
# save_model(model, "pretrained/pretrained.pth")