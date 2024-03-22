# New framework

[Chinese](README_cn.md) | English

# How to run

```
cd ~/rural-road-division-model
python train.py
```

# How to place dataset

The data set is stored in the `/data` path by default. The clean data is stored in `/data/clean`, and the unclean data is stored in `/data/raw`. Pay attention to whether the image and mask in the clean data need to correspond one by one

# Global Config

**CUDA Acceleration**

You can turn on or off CUDA acceleration for model training or inference

```config.json
{
    "cuda": true
}
```

**Log Settings**

You can set the log root directory and the number of models automatically saved per training session

```config.json
{
    "log": {
    "root": "log/",
    "save_num": 4
    }
}
```

**Data set Catalog**

You can set the save root of the dataset and download or not options

```config.json
{
    "data": {
    "download": true,
    "root": "data/"
    }
}
```

**Data loading Settings**

You can set the amount of data the data loader loads at a time

```config.json
    {
    "dataloader": {
    "train_batch_size": 1024,
    "test_batch_size": 1024
    }
}
```

**Training Settings**

You can set the training batch and model learning rate

```config.json
{
    "train": {
    "epochs": 10,
    "learning_rate": 1e-3
    }
}
```

**Visual Settings**

You can set the path to save the final training data visualization

You can set the training batch and model learning rate

```config.json
{
    "show": {
    "acc_img": "acc_img.jpg",
    "loss_img": "loss_img.jpg"
    }
}
```