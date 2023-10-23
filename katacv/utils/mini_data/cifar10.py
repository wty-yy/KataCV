import numpy as np
from typing import Tuple
import pickle
from pathlib import Path
# Download CIFAR-10 python version: https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz

def unpickle(path):
  with open(path, 'rb') as file:
    data = pickle.load(file, encoding='bytes')
  return data

def load_cifar10(path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  path = Path(path)
  assert(path.exists())
  train_images = []
  train_labels = []
  for i in range(5):
    path_train_batch = path.joinpath(f"data_batch_{i+1}")
    assert(path_train_batch.exists())
    data = unpickle(path_train_batch)
    train_images.append(np.array(data[b'data']).reshape(-1, 3, 32, 32))
    train_labels.append(np.array(data[b'labels']))
  train_images = np.concatenate(train_images, axis=0)
  train_labels = np.concatenate(train_labels, axis=0)
    
  data = unpickle(path.joinpath("test_batch"))
  test_images = np.array(data[b'data']).reshape(-1, 3, 32, 32)  # N,C,W,H
  test_labels = np.array(data[b'labels'])

  train_images = np.transpose(train_images, (0,2,3,1)).astype('float32') / 255
  test_images = np.transpose(test_images, (0,2,3,1)).astype('float32') / 255
  return train_images, train_labels, test_images, test_labels

if __name__ == '__main__':
  path = Path("/home/yy/Coding/datasets/cifar10")
  data = load_cifar10(path)
  from katacv.utils.mini_data.label2readable import label2readable
  label2readable = label2readable['cifar10']
  # data_dict = unpickle(path.joinpath("batches.meta"))
  # readables = [label.decode() for label in data_dict[b'label_names']]
  # for i, x in enumerate(readables):
  #   print(f"{i}: '{x}',")
  import matplotlib.pyplot as plt
  train_x, train_y = data[0], data[1]
  print(train_x.shape)
  for i in range(100):
    plt.imshow(train_x[i])
    plt.title(label2readable[train_y[i]])
    plt.show()


