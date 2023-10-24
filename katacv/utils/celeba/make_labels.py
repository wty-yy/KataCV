from katacv.utils.related_pkgs.jax_flax_optax_orbax import *
from katacv.utils.related_pkgs.utility import *
import numpy as np

def load_attribute(path):
  """
  Load `list_attr_celeba.txt` to a numpy boolean array.
  -1 -> False, 1 -> True
  """
  path = Path(path)
  assert(path.exists())
  attr = []
  with open(path, 'r') as file:
    n = int(file.readline())
    classes = file.readline().strip().split(' ')
    label2name = dict(enumerate(classes))
    name2label = {x: idx for (idx, x) in enumerate(classes)}
    for line in file.readlines():
      line = [False if x.strip() == '-1' else True for x in line.strip().split(' ') if x != '']
      # print(line, len(line))
      attr.append(np.array(line[1:])[None,:])
  attr = np.concatenate(attr, axis=0)
  # print(n)
  # print(classes, len(classes))
  # print(attr.shape, attr.dtype)
  return attr, label2name, name2label

def write_annotation_with_target_label(path, target_labels, train_rate=0.8):
  # basees = {x: 2**idx for (idx, x) in enumerate(target_labels)}
  bases = [name2label[label] for label in target_labels]
  print(target_labels)
  attr_slice = attr[:, bases]
  # print(attr_slice)
  labels = np.zeros((attr.shape[0],), dtype=np.int32)
  for i in range(len(target_labels)):
    labels += attr_slice[:,i] * (2 ** i)
  path = Path(path)
  train_data_size = int(attr.shape[0] * train_rate)
  val_data_size = attr.shape[0] - train_data_size
  path_image_prefix = "./img_align_celeba/"
  with open(path.joinpath("train_annotation.txt"), 'w') as file:
    for i in range(train_data_size):
      line = path_image_prefix + f"{i+1:06}.jpg {labels[i]}\n"
      file.write(line)
  print("Complete write `train_annotation.txt`: size", train_data_size)
  with open(path.joinpath("val_annotation.txt"), 'w') as file:
    for i in range(train_data_size, attr.shape[0]):
      line = path_image_prefix + f"{i+1:06}.jpg {labels[i]}\n"
      file.write(line)
  print("Complete write `val_annotation.txt`: size", val_data_size)
  return labels

def write_label2readable(target_labels, target_labels_opp):
  path = Path(__file__).parent.joinpath("label2readable.py")
  with open(path, 'w') as file:
    file.write("label2readable = {\n")
    for i in range((1 << (len(target_labels)))):
      name = ""
      for j in range(len(target_labels)):
        if i & (1<<j) == 0:
          name += target_labels_opp[j] + '_'
        else:
          name += target_labels[j] + '_'
      name = name[:-1]
      file.write(f"  {i}: '{name}',\n")
    file.write("}")

if __name__ == '__main__':
  np.random.seed(42)
  path = Path("/home/yy/Coding/datasets/celeba/")
  attr, label2name, name2label = load_attribute(path.joinpath("list_attr_celeba.txt"))
  target_labels = ['Male', 'Smiling']
  target_labels_opp = ['Female', 'noSmiling']
  labels = write_annotation_with_target_label(path, target_labels)
  write_label2readable(target_labels, target_labels_opp)
  import matplotlib.pyplot as plt
  plt.hist(labels, bins=10)
  plt.show()
