# -*- coding: utf-8 -*-
'''
@File  : vae.py
@Time  : 2023/10/22 15:01:46
@Author  : wty-yy
@Version : 1.0
@Blog  : https://wty-yy.space/
@Desc  : 
Implement VAE in JAX and train on MNIST dataset.
'''
import os, sys
sys.path.append(os.getcwd())

from katacv.utils.related_pkgs.jax_flax_optax_orbax import *
from katacv.utils.related_pkgs.utility import *

from katacv.G_VAE.model import TrainState
@partial(jax.jit, static_argnames=['train'])
def model_step(
  state: TrainState,
  x: jax.Array,
  y: jax.Array,
  train: bool
):
  def loss_fn(params):
    (distrib, logits), updates = state.apply_fn(
      {'params': params, 'batch_stats': state.batch_stats},
      x, train=train, rngs={'sample_key': state.sample_key},
      mutable=['batch_stats']
    )
    mu, logsigma2 = distrib
    loss_kl = 0.5 * (
      jnp.exp(logsigma2) - logsigma2 - 1
      + mu ** 2
    ).sum(-1).mean()
    loss_img = ((x - logits) ** 2).sum((1,2,3)).mean()
    # loss_img = -(  # use bce loss
    #   x * jax.nn.log_sigmoid(logits)
    #   + (1-x) * jax.nn.log_sigmoid(-logits)
    # ).sum((1,2,3)).mean()
    loss = loss_img + args.coef_kl_loss * loss_kl
    return loss, (updates, loss_img, loss_kl)
  
  if train:
    (loss, (updates, *metrics)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates['batch_stats'])
  else:
    loss, (_, *metrics) = loss_fn(state.params)
  return state, (loss, *metrics)

if __name__ == '__main__':
  ### Initialize arguments and tensorboard writer ###
  from katacv.G_VAE.parser import get_args_and_writer
  # args, writer = get_args_and_writer(model_name='VAE', dataset_name='MNIST')
  args, writer = get_args_and_writer(model_name='VAE', dataset_name='cifar10')

  ### Initialize log manager ###
  from katacv.G_VAE.logs import logs

  ### Initialize model state ###
  from katacv.G_VAE.model import get_vae_model_state
  state = get_vae_model_state(args)

  ### Load weights ###
  from katacv.utils import load_weights
  state = load_weights(state, args)

  ### Save config ###
  from katacv.utils import SaveWeightsManager
  save_weight = SaveWeightsManager(args)

  ### Initialize dataset ###
  from katacv.utils.mini_data.build_dataset import DatasetBuilder
  if args.path_dataset.name == 'mnist':
    from katacv.utils.mini_data.mnist import load_mnist
    data = load_mnist(args.path_dataset)
  elif args.path_dataset.name == 'cifar10':
    from katacv.utils.mini_data.cifar10 import load_cifar10
    data = load_cifar10(args.path_dataset)
  ds_builder = DatasetBuilder(data, args)
  train_ds, train_ds_size = ds_builder.get_dataset('train', repeat=args.repeat, use_aug=args.flag_use_aug)
  val_ds, val_ds_size = ds_builder.get_dataset('val')

  ### Train and evaluate ###
  start_time, global_step = time.time(), 0
  if args.train:
    for epoch in range(state.step//train_ds_size+1, args.total_epochs+1):
      print(f"epoch: {epoch}/{args.total_epochs}")
      print("training...")
      logs.reset()
      for x, y in tqdm(train_ds, total=train_ds_size):
        x, y = x.numpy(), y.numpy()
        global_step += 1
        state, metrics = model_step(state, x, y, train=True)
        logs.update(
          ['loss_train', 'loss_img_train', 'loss_kl_train'],
          metrics
        )
        if global_step % args.write_tensorboard_freq == 0:
          logs.update(
            ['SPS', 'SPS_avg', 'epoch', 'learning_rate'],
            [
              args.write_tensorboard_freq/logs.get_time_length(),
              global_step/(time.time()-start_time),
              epoch,
              args.learning_rate_fn(state.step),
            ]
          )
          logs.writer_tensorboard(writer, global_step)
          logs.reset()
      print("validating...")
      logs.reset()
      for x, y in tqdm(val_ds, total=val_ds_size):
        x, y = x.numpy(), y.numpy()
        _, loss = model_step(state, x, y, train=False)
        logs.update(
          ['loss_val', 'loss_img_val', 'loss_kl_val', 'epoch', 'learning_rate'],
          [*metrics, epoch, args.learning_rate_fn(state.step)]
        )
      logs.writer_tensorboard(writer, global_step)
      
      ### Save weights ###
      if epoch % args.save_weights_freq == 0:
        save_weight(state)
  writer.close()
