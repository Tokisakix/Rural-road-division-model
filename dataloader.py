from torch.utils.data import DataLoader

from load_config import load_config
from data import get_dataset

def get_dataloader(CONFIG, dataset, clean):
    CONFIG = load_config()
    LOADER_CONFIG = CONFIG["dataloader"]

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=LOADER_CONFIG["clean_batch_size"] if clean else LOADER_CONFIG["raw_batch_size"],
    )

    return dataloader




# ---Test---

if __name__ == "__main__":
    CONFIG = load_config()
    
    clean_dataset = get_dataset(CONFIG, clean=True)
    raw_dataset   = get_dataset(CONFIG, clean=False)

    clean_dataloader = get_dataloader(CONFIG, clean_dataset, clean=True)
    raw_dataloader   = get_dataloader(CONFIG, raw_dataset,  clean=False)

    clean_inputs, clean_labels, clean = clean_dataloader.__iter__()._next_data()
    raw_inputs, raw = raw_dataloader.__iter__()._next_data()

    print(clean_inputs.shape, clean_labels.shape, clean.shape)
    print(raw_inputs.shape, raw.shape)