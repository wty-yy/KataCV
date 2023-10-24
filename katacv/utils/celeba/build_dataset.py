# -*- coding: utf-8 -*-
'''
@File    : build_dataset.py
@Time    : 2023/10/24 15:15:05
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    :
Download CelebA: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
- `img_align_celeba.zip`: dataset
- `list_attr_celeba.txt`: make target labels annotation use `make_labels.py`

First you should use `make_labels.py` to get
`train_annotation.txt` and `val_annotation.txt`.
'''
from katacv.utils.related_pkgs.jax_flax_optax_orbax import *
from katacv.utils.related_pkgs.utility import *
import tensorflow as tf
import numpy as np

def resize(image, label, image_shape):
  image = tf.image.resize([image], image_shape)[0]
  return image, label

def augmentation(image, label):
  image = tf.image.random_flip_left_right(image)
  return image, label

def decode_image_path(path, label, image_size):
  bytes = tf.io.read_file(path)
  image = tf.io.decode_jpeg(bytes, channels=3)
  image = tf.cast(image, tf.float32) / 255
  image = tf.image.resize([image], image_size)[0]
  return image, label

class DatasetBuilder():
  train_data: Tuple[np.ndarray]  # (path_image, label)
  val_data: Tuple[np.ndarray]  # (path_image, label)
  batch_size: int
  shuffle_size: int
  image_shape: Tuple[int]  # if augmentation

  def __init__(self, args):
    self.train = load_data(args.path_dataset.joinpath("train_annotation.txt"))
    self.val = load_data(args.path_dataset.joinpath("val_annotation.txt"))
    self.batch_size, self.shuffle_size = args.batch_size, args.shuffle_size
    self.image_shape = args.image_shape[:2]
  
  def get_dataset(self, subset='train', repeat=1, shuffle=True, use_aug=False):
    assert(subset in ['train', 'val'])
    data = self.train if subset == 'train' else self.val
    datasize = data[0].shape[0]
    # return tf.data.Dataset.from_tensor_slices(data)
    decode = partial(decode_image_path, image_size=self.image_shape)
    ds = tf.data.Dataset.from_tensor_slices(data).map(decode)
    if use_aug:
      ds = ds.map(augmentation)
    ds = ds.repeat(repeat)
    if shuffle: ds = ds.shuffle(self.shuffle_size)
    ds = ds.batch(self.batch_size, drop_remainder=True)
    return ds, datasize * repeat // self.batch_size

def load_data(path_annotation):
  path = Path(path_annotation)
  assert(path.exists())
  path_images, labels = [], []
  with open(path, 'r') as file:
    for line in file.readlines():
      path_image, label = line.strip().split(' ')
      path_images.append(path.parent.joinpath(path_image))
      labels.append(int(label))
  return np.array(path_images, dtype=np.str_), np.array(labels, dtype=np.int32)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--path-dataset", type=cvt2Path, default=Path("/home/yy/Coding/datasets/celeba"))
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--shuffle-size", type=int, default=64*16)
  parser.add_argument("--image-shape", nargs='+', default=(208, 176, 3))
  args = parser.parse_args()

  ds_builder = DatasetBuilder(args)
  ds, ds_size = ds_builder.get_dataset(subset='train')
  import matplotlib.pyplot as plt
  from label2readable import label2readable
  for x, y in ds:
    x, y = x.numpy(), y.numpy()
    print(x.shape, y.shape)
    plt.imshow(x[0])
    plt.title(label2readable[y[0]])
    plt.show()

