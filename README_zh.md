# Rural-road-division-model

[English](README.md) | [中文](README_zh.md)

**文件架构**

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

**环境需求**

```requirements.txt
python==3.11.6
gradio==4.1.2
gradio_client==0.7.0
numpy==1.26.1
opencv-python==4.8.1.78
torch==2.1.0+cu118
```

**如何使用该框架**

1. 此框架已经提前写好数据集加载程序, 你可以通过 **get_loader()** 函数来获取数据加载器, 你可以在 **config.json** 中填写你需要数据集的信息, 数据加载器的使用案例和接口如下:

```python
from utils.loader import get_loader

train_loader, test_loader = get_loader()

# train_loader 有三个返回值，类型均为 FloatTensor, 分别代表原始图片，原始图片的标注，负例图片
# - original image           : FloatTensor[batch_size, 3, 1024, 1024]
# - mask                     : FloatTensor[batch_size, 1, 1024, 1024]
# - negative example image   : FloatTensor[batch_size, 3, 1024, 1024]
#
# test_loader 有两个返回值，类型均为 FloatTensor, 分别代表原始图片，原始图片的标注
# - original image           : FloatTensor[batch_size, 3, 1024, 1024]
# - mask                     : FloatTensor[batch_size, 1, 1024, 1024]
```

2. 这个框架允许你自定义你的模型，只要保证你的模型数据接口与下列一致

```python
from utils.model import Model

model = Model()
# model.train() 接受三个参数，并返回本次训练的损失值
# [input]   original image   : FloatTensor[batch_size, 3, 1024, 1024]
# [input]   positive example : FloatTensor[batch_size, 3, 1024, 1024]
# [input]   negtive example  : FloatTensor[batch_size, 3, 1024, 1024]
# [output]  train loss       : float
#
# model.infer() 接受一个参数并返回模型的预测结果
# [input]   original image   : FloatTensor[batch_size, 3, 1024, 1024]
# [output]  predict image    : FloatTensor[batch_size, 3, 1024, 1024]
```

**数据集**

你可以将你的数据集放在路径 "/data/{your dataset name}/" 中
并在 config.json 中添加下列属性

- **name**: 数据集的名称
- **use**: 确认是否使用
- **train_root**: 训练集的路径
- **test_root**: 测试集的路径
- **image_**: 原始图片的后缀
- **mask_**: 标注图片的后缀

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

**模型**

你可以在 config.json 中设置模型的属性

```json
{
"model":{
        "PartC":{
            "img_size" : 256,
            "patch_size" : 16,
            "in_channels" : 3,
            "embed_dim" : 256,
            "num_heads" : 2,
            "num_layers" : 2,
            "mlp_hidden_dim" : 128,
            "classify_dim" : 128,
            "output_token_size" : 128
        }
    },
}
```

**训练**

你可以输入下列代码开始训练模型

```bash
python train.py
```

你可以在 config.json 中设置本次训练的属性

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

**日志**

你可以在 config.json 中设置日志的属性

```json
{
    "log":{
        "root":"log/"
    }
}
```

**Webui**

你需要输入下列代码来启动你的 webui

```bash
python train.py
```

你可以在 config.json 中设置 webui 的相关信息

```json
{
    "webui":{
        "share":false,
        "port":8080,
        "model_path":"models/model.pth"
    }
}
```
