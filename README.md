# Rural-road-division-model

**File architecture**

```tree
Rural-road-division-model:.
├─data
│  ├─croproad_dataset
│  │  ├─test
│  │  └─train
│  └─deepglobal
│      ├─test
│      ├─train
│      └─valid
├─flagged
├─log
│  ├─ViT_v0
│  └─ViT_v1
├─models
├─network
├─pretrained
└─utils
    ├─criterion
    ├─loader
    ├─logger
    ├─model
```

**Requirements**

```requirements.txt
python==3.11.6
gradio==4.1.2
gradio_client==0.7.0
numpy==1.26.1
opencv-python==4.8.1.78
torch==2.1.0+cu118
```

**How to use this framework**

1. The framework automatically writes the interface of the dataloader in advance, you can get the dataloader through the function **get_loader()**, you can set the parameter information of the dataloader and the information of the dataset in **config.json**, the interface information is as follows:

```python
from utils.loader import get_loader

train_loader, test_loader = get_loader()

# train_loader return three FloatTensor, the original image, the mask and the negative example image
# - original image           : FloatTensor[batch_size, 3, 1024, 1024]
# - mask                     : FloatTensor[batch_size, 1, 1024, 1024]
# - negative example image   : FloatTensor[batch_size, 3, 1024, 1024]
#
# test_loader return two FloatTensor, the original image and the mask
# - original image           : FloatTensor[batch_size, 3, 1024, 1024]
# - mask                     : FloatTensor[batch_size, 1, 1024, 1024]
```

2. The framework allows you to design your model freely, make sure that the input and output interfaces of your model are as follows:

```python
from utils.model import Model

model = Model()
# model.train() accept three parameters and return the loss value for this training
# [input]   original image   : FloatTensor[batch_size, 3, 1024, 1024]
# [input]   positive example : FloatTensor[batch_size, 3, 1024, 1024]
# [input]   negtive example  : FloatTensor[batch_size, 3, 1024, 1024]
# [output]  train loss       : float
#
# model.infer() accept one parameters and return the predict image
# [input]   original image   : FloatTensor[batch_size, 3, 1024, 1024]
# [output]  predict image    : FloatTensor[batch_size, 3, 1024, 1024]
```

**Data**

you can put your dataset in the path "/data/{your dataset name}/"
and add the dataset's information in the config.json

- **name**: Dataset's name
- **use**: Confirm whether to use
- **train_root**: Train dataset's root
- **test_root**: Test dataset's root
- **image_**: Image file suffix
- **mask_**: Mask file suffix

```json
{
    "data":[
        {
            "name":"croproad",
            "use":true,
            "train_root":"data/croproad_dataset/train/",
            "test_root":"data/croproad_dataset/test/",
            "image_":"_1.png",
            "mask_":"_1 (2).png"
        },
        {
            "name":"deepglobal",
            "use":true,
            "train_root":"data/deepglobal/train/",
            "test_root":"data/deepglobal/test/",
            "image_":"_sat.jpg",
            "mask_":"_mask.png"
        }
    ]
}
```

**Train**

to train the Rural-road-division-model, you can enter

```bash
python train.py
```

you can set the train config in the config.json

```json
{
    "train":{
        "name":"ViT_v1",
        "pretrained":false,
        "train_batch_size":4,
        "test_batch_size":4,
        "train_epochs":10,
        "learning_rate":1e-3,
        "show_info_step":8,
        "test_info_step":8,
        "save_model_step":8,
        "save_model_num":4
    }
}
```

**Log**

you can set the log config in the config.json

```json
{
    "log":{
        "root":"log/"
    }
}
```

**Webui**

to use the webui, you can enter

```bash
python train.py
```

you can set the webui config in the config.json

```json
{
    "webui":{
        "share":false,
        "port":8080,
        "model_path":"models/model.pth"
    }
}
```
