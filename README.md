# Rural-road-division-model

## data
you can put your dataset in the path "/data/{your dataset name}/"
and add the dataset's information in the config.json

```
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

## train
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

## webui
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
