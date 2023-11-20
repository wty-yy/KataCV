from pathlib import Path
import jax.numpy as jnp

path_darknet_weights = Path("/home/wty/Coding/models/YOLOv4/CSPDarkNet53-0050-lite")

dataset_name = 'COCO'  # or 'PASCAL'
if dataset_name == 'COCO':
  # path_dataset = Path("/media/yy/Data/dataset/COCO")
  path_dataset = Path("/home/wty/Coding/datasets/coco")
  num_classes = 80
  train_ds_size = 118287
  use_mosaic4 = False
elif dataset_name == 'PASCAL':
  path_dataset = Path("/media/yy/Data/dataset/PASCAL")
  num_classes = 20
  train_ds_size = 16551
  use_mosaic4 = False
num_data_workers = 6

image_shape = (608, 608, 3)  # input shape: 320 + 96 * n
# image_shape = (416, 416, 3)  # input shape: 320 + 96 * n
anchors = jnp.array([  # Specify pixels, shape: (3, 3, 2)
  [[12, 16], [19, 36], [40, 28]],
  [[36, 75], [76, 55], [72, 146]],
  [[142, 110], [192, 243], [459, 401]]
])

### Training ###
total_epochs = 100
batch_size = 64
learning_rate = 0.05
weight_decay = 1e-4
warmup_epochs = 3
momentum = 0.9  # 'sgd'