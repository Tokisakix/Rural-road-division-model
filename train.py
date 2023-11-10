from utils import load_config, get_basic_dataset

config = load_config()
data_config = config["data"]

basic_train_set, basic_test_set = get_basic_dataset(data_config)