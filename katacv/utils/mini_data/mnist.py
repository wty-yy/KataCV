import gzip
import os
from urllib.request import urlretrieve
import numpy as np
from typing import Tuple

# reference: https://mattpetersen.github.io/load-mnist-with-numpy
def load_mnist(path=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  url = 'http://yann.lecun.com/exdb/mnist/'
  files = ['train-images-idx3-ubyte.gz',
       'train-labels-idx1-ubyte.gz',
       't10k-images-idx3-ubyte.gz',
       't10k-labels-idx1-ubyte.gz']

  if path is None:
    # Set path to /home/USER/data/mnist or C:\Users\USER\data\mnist
    path = os.path.join(os.path.expanduser('~'), 'data', 'mnist')

  # Create path if it doesn't exist
  os.makedirs(path, exist_ok=True)

  # Download any missing files
  for file in files:
    if file not in os.listdir(path):
      urlretrieve(url + file, os.path.join(path, file))
      print("Downloaded %s to %s" % (file, path))

  def _images(path):
    """Return images loaded locally."""
    with gzip.open(path) as f:
      # First 16 bytes are magic_number, n_imgs, n_rows, n_cols
      pixels = np.frombuffer(f.read(), 'B', offset=16)
    return pixels.reshape(-1, 28, 28, 1).astype('float32') / 255

  def _labels(path):
    """Return labels loaded locally."""
    with gzip.open(path) as f:
      # First 8 bytes are magic_number, n_labels
      integer_labels = np.frombuffer(f.read(), 'B', offset=8)
    return integer_labels

  train_images = _images(os.path.join(path, files[0]))
  train_labels = _labels(os.path.join(path, files[1]))
  test_images = _images(os.path.join(path, files[2]))
  test_labels = _labels(os.path.join(path, files[3]))

  return train_images, train_labels, test_images, test_labels 