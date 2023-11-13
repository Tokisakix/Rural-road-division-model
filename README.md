# Rural-road-division-model

## File architecture

```
T:.
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
├─pretrained
└─utils
    ├─criterion
    ├─loader
    ├─logger
    ├─model
```

## Requirements

```requirements.txt
python==3.11.6
gradio==4.1.2
gradio_client==0.7.0
numpy==1.26.1
opencv-python==4.8.1.78
torch==2.1.0+cu118
```

## Data

you can put your dataset in the path "/data/{your dataset name}/"
and add the dataset's information in the config.json

- **name**: Dataset's name
- **use**: Confirm whether to use
- **train_root**: Train dataset's root
- **test_root**: Test dataset's root
- **image_**: Image file suffix
- **mask_**: Mask file suffix

```config.json
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

## Train

to train the Rural-road-division-model, you can enter

```bash
python train.py
```

you can set the train config in the config.json

```config.json
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

## Log

you can set the log config in the config.json

```config.json
{
    "log":{
        "root":"log/"
    }
}
```

## Webui

to use the webui, you can enter

```bash
python train.py
```

you can set the webui config in the config.json

```config.json
{
    "webui":{
        "share":false,
        "port":8080,
        "model_path":"models/model.pth"
    }
}
```
