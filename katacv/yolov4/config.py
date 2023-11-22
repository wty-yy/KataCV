from pathlib import Path
import jax.numpy as jnp

path_darknet_weights = Path("/home/yy/Coding/models/YOLOv4/CSPDarkNet53-0050-lite")

dataset_name = 'COCO'  # or 'PASCAL'
# dataset_name = 'PASCAL VOC'  # or 'PASCAL'
if dataset_name == 'COCO':
  # path_dataset = Path("/home/wty/Coding/datasets/coco")
  path_dataset = Path("/home/yy/Coding/datasets/coco")
  # path_dataset = Path('/media/yy/Data/dataset/COCO')
  num_classes = 80
  train_ds_size = 118287
  # train_ds_size = 800  # sample test
  use_mosaic4 = False
if dataset_name == 'PASCAL VOC':
  path_dataset = Path("/home/yy/Coding/datasets/PASCAL")
  # path_dataset = Path("/home/wty/Coding/datasets/PASCAL")
  num_classes = 20
  train_ds_size = 16550
  use_mosaic4 = False
num_data_workers = 8

# image_shape = (608, 608, 3)  # input shape: 320 + 96 * n
image_shape = (416, 416, 3)  # input shape: 320 + 96 * n
anchors = jnp.array([  # Specify pixels, shape: (3, 3, 2)
  [[12, 16], [19, 36], [40, 28]],
  [[36, 75], [76, 55], [72, 146]],
  [[142, 110], [192, 243], [459, 401]]
])

### Training ###
if dataset_name == 'COCO':
  batch_size = 32
  total_epochs = 300
if dataset_name == 'PASCAL VOC':
  batch_size = 32
  total_epochs = 100
coef_noobj = 2.0
coef_coord = 2.0
coef_obj = 1.0
coef_class = 1.0
base_learning_rate = 2.5e-4
# base_learning_rate = 1e-3
learning_rate = base_learning_rate * batch_size / 256
weight_decay = 1e-4
warmup_epochs = 2
momentum = 0.9  # if optimizer is SGD