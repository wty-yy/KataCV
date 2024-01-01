from pathlib import Path
import jax.numpy as jnp

path_darknet_weights = Path("/home/yy/Coding/models/YOLOv5/NewCSPDarkNet53-0050-lite")

dataset_name = 'COCO'  # or 'PASCAL'
# dataset_name = 'PASCAL VOC'  # or 'PASCAL'
if dataset_name == 'COCO':
  # path_dataset = Path("/home/wty/Coding/datasets/coco")
  path_dataset = Path("/home/yy/Coding/datasets/coco")
  num_classes = 80
  train_ds_size = 118287
  # train_ds_size = 800  # sample test
if dataset_name == 'PASCAL VOC':
  path_dataset = Path("/home/yy/Coding/datasets/PASCAL")
  # path_dataset = Path("/home/wty/Coding/datasets/PASCAL")
  num_classes = 20
  train_ds_size = 16550
num_data_workers = 8

use_mosaic4 = True
hsv_h = 0.015  # HSV-Hue augmentation
hsv_s = 0.7  # HSV-Saturation augmentation
hsv_v = 0.4  # HSV-Value augmentation
translate = 0.1  # translation (+/- fraction)
scale = 0.5  # scale (+/- gain)
fliplr = 0.5  # flip left-right (probability)

# image_shape = (416, 416, 3)
image_shape = (640, 640, 3)
anchors = jnp.array([
  [[10,13], [16,30], [33,23]],  # P3/8
  [[30,61], [62,45], [59,119]],  # P4/16
  [[116,90], [156,198], [373,326]]  # P5/32
])

### Training ###
if dataset_name == 'COCO':
  batch_size = 32
  total_epochs = 300
if dataset_name == 'PASCAL VOC':
  batch_size = 32
  total_epochs = 100
coef_box = 0.05
coef_obj = 1.0
coef_cls = 0.5
# coef_batch = batch_size / 64
# learning_rate_init = 0.01 * coef_batch
# learning_rate_final = 1e-4 * coef_batch
# weight_decay = 5e-4 * coef_batch
learning_rate_init = 0.01
learning_rate_final = 1e-4
weight_decay = 5e-4
warmup_epochs = 3
momentum = 0.937  # if optimizer is SGD

if __name__ == '__main__':
  from pathlib import Path
  print(Path(__file__).resolve().parent)
