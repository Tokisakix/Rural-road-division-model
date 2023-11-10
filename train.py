from utils import load_config
from utils.loader import get_loader
from utils.model import Model, load_model, save_model
from utils.criterion import DicBceLoss

CONFIG      = load_config()
DATA_CONFIG = CONFIG["data"]
CUDA        = CONFIG["cuda"]

# model = Model().cuda() if CUDA else Model()
model = load_model("pretrained/pretrained.pth")

train_loader, test_loader = get_loader()

for image, pos, neg in train_loader:
    print(image.shape, pos.shape, neg.shape)
    break

for image, mask in test_loader:
    print(image.shape, mask.shape)
    break

# save_model(model, "models/model.pth")
# save_model(model, "pretrained/pretrained.pth")