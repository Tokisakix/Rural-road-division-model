from utils import load_config
from utils.loader import get_basic_dataset
from utils.model import ViT
from utils.criterion import DicBceLoss

config = load_config()
data_config = config["data"]

basic_train_set, basic_test_set = get_basic_dataset(data_config)

inputs, labels = basic_train_set[0]
inputs, labels = inputs.unsqueeze(0), labels.unsqueeze(0)
outputs = ViT()(inputs)

print(inputs.shape, labels.shape, outputs.shape)

loss = DicBceLoss()(outputs, labels)

print(loss)