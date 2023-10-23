# -*- coding: utf-8 -*-
'''
@File  : vae.py
@Time  : 2023/10/22 16:01:46
@Author  : wty-yy
@Version : 1.0
@Blog  : https://wty-yy.space/
@Desc  : 
2023/10/22: Predict G-VAE output.
2023/10/23: Fine-tuning:
1. clip variance by rate: 0.05, 0.1, 0.3
Optimize:
1. change the output of the image layout
'''
import os, sys
sys.path.append(os.getcwd())

from katacv.utils.related_pkgs.jax_flax_optax_orbax import *
from katacv.utils.related_pkgs.utility import *
import matplotlib.pyplot as plt
import numpy as np

from katacv.G_VAE.model import TrainState
@jax.jit
def predict(
  state: TrainState,
  x: jax.Array,
):
  return state.apply_fn(
    {'params': state.params, 'batch_stats': state.batch_stats},
    x, train=False
  )

def get_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--subset", default='val')
  parser.add_argument("--row", type=int, default=5)
  parser.add_argument("--column", type=int, default=5)
  return parser.parse_args([])

def show_orgin_pred():
  for x, y in tqdm(ds, total=ds_size):
    fig, axs = plt.subplots(pred_args.row, pred_args.column, figsize=(pred_args.column*4, pred_args.row*4))
    x, y = x.numpy(), y.numpy()
    distrib, pred_x, logits = jax.device_get(predict(state, x))
    pred_x = np.clip(pred_x, 0.0, 1.0)
    # print(logits[0].min(), logits[0].max(), logits[0].mean())
    for i in range(pred_args.row):
      for j in range(pred_args.column):
        ax = axs[i, j]
        idx = i * pred_args.column + j
        org, pred = x[idx], pred_x[idx]
        image = np.concatenate([org, pred], axis=1)
        ax.imshow(image)
        ax.set_title(f"predict: {int(np.argmax(logits[idx]))}")
        ax.set_axis_off()
    plt.tight_layout()
    plt.show()

from PIL import Image
def show_image_change(x1, x2, n=10, name="image_change"):
  image = x1  # (B,N,N,1)
  distrib1, _, _ = jax.device_get(predict(state, x1))
  distrib2, _, _ = jax.device_get(predict(state, x2))
  mu1, mu2 = distrib1[0], distrib2[0]
  delta = (mu2 - mu1) / n
  for _ in range(n):
    logits = jax.device_get(predict(decoder_state, mu1))
    # logits = (logits - logits.min()) / (logits.max() - logits.min())
    logits = np.clip(logits, 0, 1)
    # print(image.shape, logits.shape)
    image = np.concatenate([image, logits], axis=2)
    mu1 = mu1 + delta
  image = np.concatenate([image, x2], axis=2)
  image = image.reshape((-1, *image.shape[-2:]))
  image = (image[..., 0] * 255).astype('uint8')
  image = Image.fromarray(image)
  image.save(str(pred_args.path_figures.joinpath(name+'.jpg')))
  image.show()

def show_image_aug(x, n=10, name='image_aug', threshold_rate=0.1):
  distrib, _, _ = jax.device_get(predict(state, x))
  mu, logsigma2 = distrib
  sigma = np.sqrt(np.exp(logsigma2))
  # print(np.sort(sigma, axis=-1)[0])
  threshold_idx = int(sigma.shape[1]*(1-threshold_rate))
  threshold = np.sort(sigma, axis=-1)[:, threshold_idx:threshold_idx+1]
  delta = np.where(sigma >= threshold, sigma, 0)
  plt.hist(sigma[0], bins=50)
  plt.hist(delta[0][delta[0] > 0], bins=50)
  plt.show()
  # delta = delta / (delta**2).sum(-1)[:,None]
  # delta = sigma / (sigma ** 2).sum(-1)[:,None]
  # delta = sigma / sigma.sum(-1)[:,None]
  print(mu.mean(), sigma.mean())
  pos, neg = [], []
  for i in range(n//2):
    z = mu - i * delta * 0.5
    aug = jax.device_get(predict(decoder_state, z))
    aug = (aug - aug.min()) / (aug.max() - aug.min())
    # aug = np.clip(aug, 0, 1)
    neg.append(aug)
    # image = np.concatenate([image, aug], axis=2)

    z = mu + i * delta * 0.5
    aug = jax.device_get(predict(decoder_state, z))
    aug = (aug - aug.min()) / (aug.max() - aug.min())
    # aug = np.clip(aug, 0, 1)
    pos.append(aug)
    # image = np.concatenate([image, aug], axis=2)
  image = 1 - x  # mid: (B,N,N,1)
  for aug in neg: image = np.concatenate([aug, image], axis=2)
  for aug in pos: image = np.concatenate([image, aug], axis=2)
  image = image.reshape((-1, *image.shape[-2:]))
  image = (image[..., 0]*255).astype('uint8')
  image = Image.fromarray(image)
  image.save(str(pred_args.path_figures.joinpath(name+'.jpg')))
  image.show()

if __name__ == '__main__':
  ### Initialize arguments and tensorboard writer ###
  from katacv.G_VAE.parser import get_args_and_writer
  vae_args = get_args_and_writer(no_writer=True)
  pred_args = get_args()
  vae_args.batch_size = pred_args.row * pred_args.column
  pred_args.path_figures = vae_args.path_logs.joinpath("figures")
  pred_args.path_figures.mkdir(exist_ok=True)

  ### Initialize model state ###
  from katacv.G_VAE.model import get_g_vae_model_state, get_decoder_state
  state = get_g_vae_model_state(vae_args)
  decoder_state = get_decoder_state(vae_args)

  ### Load weights ###
  from katacv.utils import load_weights
  state = load_weights(state, vae_args)
  decoder_state = decoder_state.replace(
    params=state.params['Decoder_0'],
    batch_stats=state.batch_stats['Decoder_0']
  )

  ### Initialize dataset ###
  from katacv.utils.mini_data.build_dataset import DatasetBuilder
  if vae_args.path_dataset.name == 'mnist':
    from katacv.utils.mini_data.mnist import load_mnist
    data = load_mnist(vae_args.path_dataset)
  ds_builder = DatasetBuilder(data, vae_args)
  ds, ds_size = ds_builder.get_dataset(pred_args.subset, shuffle=False)

  # show_orgin_pred()
  for i, (x, y) in enumerate(ds):
    if i == 3: break
    x1 = x.numpy()
    x2 = jnp.concatenate([x1[1:], x1[0:1]], axis=0)
    # show_image_change(
    #   x1[:pred_args.row],
    #   x2[:pred_args.row],
    #   n=10,
    #   name=f"image_change_{i}"
    # )
    rate = 0.1
    show_image_aug(
      x1[:10],
      n=14,
      name=f"image_aug_rate_{rate}_{i}",
      threshold_rate=rate
    )
