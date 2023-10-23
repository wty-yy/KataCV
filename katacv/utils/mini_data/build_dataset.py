# -*- coding: utf-8 -*-
'''
@File  : build_dataset.py
@Time  : 2023/10/22 11:17:01
@Author  : wty-yy
@Version : 1.0
@Blog  : https://wty-yy.space/
@Desc  : 
Load mini_data to tensorflow dataset,
mini_data refers to the datasets can be read into cache directly.

Useage:
  data = load_mnist(save_dirctory)
  ds_builder = DatasetBuilder(data, args)
  ds, ds_size = ds_builder.get_dataset(subset='train')
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

class DatasetBuilder():
  train: Tuple[np.ndarray]  # (image, label)
  val: Tuple[np.ndarray]  # (image, label)
  batch_size: int
  shuffle_size: int
  image_shape: Tuple[int]  # if augmentation

  def __init__(self, dataset, args):
    self.train = dataset[:2]
    self.val = dataset[-2:]
    self.batch_size, self.shuffle_size = args.batch_size, args.shuffle_size
    self.image_shape = args.image_shape[:2]
  
  def get_dataset(self, subset='train', repeat=1, shuffle=True, use_aug=False):
    assert(subset in ['train', 'val'])
    data = self.train if subset == 'train' else self.val
    datasize = data[0].shape[0]
    resize_fn = partial(resize, image_shape=self.image_shape)
    ds = tf.data.Dataset.from_tensor_slices(data).map(resize_fn)
    if use_aug:
      ds = ds.map(augmentation)
    ds = ds.repeat(repeat)
    if shuffle: ds = ds.shuffle(self.shuffle_size)
    ds = ds.batch(self.batch_size, drop_remainder=True)
    return ds, datasize * repeat // self.batch_size

def test_load_mnist():
  from mnist import load_mnist
  data = load_mnist("/home/yy/Coding/datasets/mnist")
  ds_builder = DatasetBuilder(data, args)
  ds, ds_size = ds_builder.get_dataset(subset='train')
  import matplotlib.pyplot as plt
  for x, y in ds:
    x, y = x.numpy(), y.numpy()
    print(x.shape, y.shape)
    plt.imshow(x[0], cmap='gray')
    plt.title(str(y[0]))
    plt.show()

def test_load_cifar10():
  from cifar10 import load_cifar10
  data = load_cifar10("/home/yy/Coding/datasets/cifar10")
  ds_builder = DatasetBuilder(data, args)
  ds, ds_size = ds_builder.get_dataset(subset='train', use_aug=True)
  import matplotlib.pyplot as plt
  for x, y in ds:
    x, y = x.numpy(), y.numpy()
    print(x.shape, y.shape)
    plt.imshow(x[0])
    plt.title(str(y[0]))
    plt.show()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--shuffle-size", type=int, default=64*16)
  parser.add_argument("--image-shape", nargs='+', default=(56, 56, 3))
  args = parser.parse_args()

  # test_load_mnist()
  test_load_cifar10()
