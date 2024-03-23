import torch, random
import numpy as np
import jax, jax.numpy as jnp
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import re
txt_convert = {
  'minus one enter': lambda text: re.sub(r'\n+', lambda match: '\n' * (len(match.group(0)) - 1), text)
}

class TextDatasetBuilder:
  def __init__(self, path_dataset, val_ratio = 0.2, seed = 42, n_divide = 100, cvt_format=None):
    np.random.seed(seed)
    torch.manual_seed(seed)
    text = ""
    paths = Path(path_dataset).glob('*.txt')
    assert Path(path_dataset).exists(), f"Can't find dataset {path_dataset}"
    for p in paths:
      with p.open('r') as file:
        text += file.read()
    if cvt_format:
      text = txt_convert[cvt_format](text)
    chars = sorted(list(set(text)))
    self.n_vocab = len(chars)
    self.idx2char = dict(enumerate(chars))
    self.char2idx = {c: i for i, c in self.idx2char.items()}
    print("Total text length:", len(text), "Vocabulary size:", self.n_vocab)
    data = self.encode(text)
    # Divide text in to 'n_divide' block, each block give to train and val in ratio
    block_size = len(data) // n_divide
    train_block_size = int(block_size * (1 - val_ratio))
    self.data = {
      'train': np.concatenate([data[i:i+train_block_size] for i in range(0, len(data), block_size)]),
      'val': np.concatenate([data[i+train_block_size:i+block_size] for i in range(0, len(data), block_size)])
    }

  def encode(self, x): return np.array([self.char2idx[c] for c in x], np.int32)
  def decode(self, x): return ''.join([self.idx2char[i] for i in x])
  
  def get_dataset(self, format='train', batch_size=128, n_token=256, datasize=500 * 512):
    return DataLoader(
      TextDataset(self.data[format], n_token, datasize),
      batch_size=batch_size,
      shuffle=format == 'train',
      num_workers=8,
      drop_last=True,
      persistent_workers=True,
    )

class TextDataset(Dataset):
  def __init__(self, data, n_token, datasize):
    self.data, self.n_token, self.datasize = data, n_token, datasize
  
  def __len__(self):
    return self.datasize
    # return len(self.data) - self.n_token
  
  def __getitem__(self, idx):
    idx = random.randint(0, len(self.data) - 2 - self.n_token)
    d = self.data[idx:idx+self.n_token+1]
    x, y = d[:self.n_token], d[1:]
    return x, y

if __name__ == '__main__':
  ds_builder = TextDatasetBuilder(path_dataset=Path("/home/yy/Coding/datasets/china_offical_documents"), cvt_format='minus one enter')
  # ds_builder = TextDatasetBuilder(path_dataset=Path("/home/yy/Coding/datasets/china_offical_documents"), cvt_format=None)
  print(f"{ds_builder.data['train'].shape=}, {ds_builder.data['val'].shape=}")
  print("Train data (first 100)")
  print(ds_builder.decode(jax.device_get(ds_builder.data['train'][:100])))
  print("Val data (first 100)")
  print(ds_builder.decode(jax.device_get(ds_builder.data['val'][:100])))
  train_ds = ds_builder.get_dataset('train')
  val_ds = ds_builder.get_dataset('val')
  print(len(train_ds), len(val_ds))
  for x, y in train_ds:
    x, y = x.numpy(), y.numpy()
    print(x.shape, y.shape)
    print("Input")
    print(ds_builder.decode(x[0]))
    print("Target")
    print(ds_builder.decode(y[0]))
    input()# break
