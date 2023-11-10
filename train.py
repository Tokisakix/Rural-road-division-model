from utils import load_config
from utils.loader import get_basic_dataset
from utils.model import ViT, load_model, save_model
from utils.criterion import DicBceLoss

CONFIG      = load_config()
DATA_CONFIG = CONFIG["data"]
CUDA        = CONFIG["cuda"]

basic_train_set, basic_test_set = get_basic_dataset(DATA_CONFIG)

inputs, labels = basic_train_set[0]
inputs, labels = inputs.unsqueeze(0), labels.unsqueeze(0)
# model = ViT().cuda() if CUDA else ViT().cuda()
model = load_model("pretrained/pretrained.pth")
outputs = model(inputs)

print(inputs.shape, labels.shape, outputs.shape)

loss = DicBceLoss()(outputs, labels)

print(loss)

save_model(model, "pretrained/pretrained.pth")