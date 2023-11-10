from torch.utils.data import DataLoader

from utils import load_config
from .dataset import get_final_set

CONFIG = load_config()
TRAIN_CONFIG = CONFIG["train"]
TRAIN_BATCH_SIZE = TRAIN_CONFIG["train_batch_size"]
TEST_BATCH_SIZE = TRAIN_CONFIG["test_batch_size"]

# 获取最终加载器
def get_loader():
    final_train_set, final_test_set = get_final_set()
    train_loader = DataLoader(final_train_set, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=0)
    test_loader = DataLoader(final_test_set, batch_size=TEST_BATCH_SIZE, shuffle=False, num_workers=0)
    return train_loader, test_loader