# -*- coding: utf-8 -*-
'''
@File    : encoder.py
@Time    : 2023/10/22 13:02:43
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : 
Encoder: Train 3 epochs on MNIST classification:
train:  loss: 0.02772, acc: 0.99275, 2s/epochs
val:    loss: 0.03308, acc: 0.99028, >1s/epochs
'''
import sys, os
sys.path.append(os.getcwd())
from typing import Any
from katacv.utils.related_pkgs.jax_flax_optax_orbax import *
from katacv.utils.related_pkgs.utility import *

class ConvBlock(nn.Module):
  filters: int
  norm: nn.Module
  act: Callable
  kernel: Tuple[int, int]
  strides: Tuple[int, int] = (1, 1)
  use_norm: bool = True
  use_act: bool = True

  @nn.compact
  def __call__(self, x):
    x = nn.Conv(self.filters, self.kernel, self.strides, use_bias=not self.use_norm)(x)
    if self.use_norm: x = self.norm()(x)
    if self.use_act: x = self.act(x)
    return x
    
class ConvTransposeBlock(nn.Module):
  filters: int
  norm: nn.Module
  act: Callable
  kernel: Tuple[int, int]
  strides: Tuple[int, int] = (1, 1)
  padding: str = 'SAME'
  use_norm: bool = True
  use_act: bool = True

  @nn.compact
  def __call__(self, x):
    x = nn.ConvTranspose(self.filters, self.kernel, self.strides, self.padding, use_bias=not self.use_norm)(x)
    if self.use_norm: x = self.norm()(x)
    if self.use_act: x = self.act(x)
    return x

class ResBlock(nn.Module):
  conv: nn.Module
  norm: nn.Module
  act: Callable

  @nn.compact
  def __call__(self, x):
    n = x.shape[-1] // 2
    residue = x
    x = self.conv(filters=n, kernel=(1,1))(x)
    x = self.conv(filters=2*n, kernel=(3,3), use_act=False)(x)
    return self.act(residue + x)

class NeckBlock(nn.Module):
  filters: int
  conv: nn.Module
  convT: nn.Module
  block: nn.Module

  @nn.compact
  def __call__(self, x):
    n = self.filters
    x = self.conv(filters=2*n, kernel=(1,1))(x)
    x = self.block()(x)
    x = self.block()(x)
    # x = self.convT(filters=2*n, kernel=(3,3), strides=(2,2))(x)
    x = jax.image.resize(x, (x.shape[0], x.shape[1]*2, x.shape[2]*2, x.shape[3]), method='nearest')
    x = self.conv(filters=n, kernel=(1,1))(x)
    return x

def mish(x):
  return x * jnp.tanh(jax.nn.softplus(x))

class Encoder(nn.Module):
  filters: int
  output_size: int
  stage_size: Sequence[int]
  act: Callable = mish

  @nn.compact
  def __call__(self, x, train: bool = True):
    norm = partial(nn.BatchNorm, use_running_average=not train)
    conv = partial(ConvBlock, norm=norm, act=self.act)
    block = partial(ResBlock, conv=conv, norm=norm, act=self.act)
    x = conv(filters=self.filters, kernel=(7,7), strides=(2,2))(x)
    scales = [x]
    for block_num in self.stage_size:
      x = conv(filters=x.shape[-1]*2, kernel=(3,3), strides=(2,2))(x)
      for _ in range(block_num):
        x = block()(x)
      scales.append(x)
    x = x.mean((1, 2))
    x = nn.Dense(self.output_size)(x)
    return x, scales

class Decoder(nn.Module):
  filters: int
  output_shape: int
  stage_length: int
  act: Callable = mish

  @nn.compact
  def __call__(self, x, scales, train: bool = True):
    norm = partial(nn.BatchNorm, use_running_average=not train)
    conv = partial(ConvBlock, norm=norm, act=self.act)
    convT = partial(ConvTransposeBlock, norm=norm, act=self.act)
    block = partial(ResBlock, conv=conv, norm=norm, act=self.act)
    neck = partial(NeckBlock, conv=conv, convT=convT, block=block)
    power = 2 ** self.stage_length
    input_shape =(self.output_shape[0] // power, self.output_shape[1] // power, self.filters)
    x = nn.Dense(input_shape[0] * input_shape[1] * input_shape[2])(x)
    x = x.reshape(x.shape[0], *input_shape)
    for i in range(self.stage_length):
      n = x.shape[-1]
      x = jnp.concatenate([x, scales[-(i+1)]], axis=-1)
      x = neck(filters=n//2)(x)
    x = conv(filters=x.shape[-1]*2, kernel=(3,3))(x)
    x = conv(filters=self.output_shape[-1], kernel=(1,1))(x)
    return x

class VAE(nn.Module):
  image_shape: Sequence[int]
  encoder_start_filters: int
  encoder_stage_size: Sequence[int]
  decoder_start_filters: int
  feature_size: int

  @nn.compact
  def __call__(self, x, train: bool = True):
    encode, scales = Encoder(
      filters=self.encoder_start_filters,
      output_size=self.feature_size * 2,
      stage_size=self.encoder_stage_size,
    )(x, train=train)
    mu, logsigma2 = encode[:, :self.feature_size], encode[:, self.feature_size:]
    distrib = (mu, logsigma2)
    z = mu
    if train:
      key = self.make_rng('sample_key')
      epsilon = jax.random.normal(key, (x.shape[0], self.feature_size))
      z = z + jnp.exp(0.5 * logsigma2) * epsilon
    decode = Decoder(
      filters=self.decoder_start_filters,
      output_shape=self.image_shape,
      stage_length=len(self.encoder_stage_size) + 1,
    )(z, scales, train=train)
    return distrib, decode, scales

class G_VAE(nn.Module):
  image_shape: Sequence[int]
  class_num: int
  encoder_start_filters: int
  encoder_stage_size: Sequence[int]
  decoder_start_filters: int
  feature_size: int

  @nn.compact
  def __call__(self, x, train: bool = True):
    encode, scales = Encoder(
      filters=self.encoder_start_filters,
      output_size=self.feature_size * 2,
      stage_size=self.encoder_stage_size,
    )(x, train=train)
    # print(len(scales))
    mu, logsigma = encode[:, :self.feature_size], encode[:, self.feature_size:]
    distrib = (mu, logsigma)
    z = mu
    if train:
      key = self.make_rng('sample_key')
      epsilon = jax.random.normal(key, (x.shape[0], self.feature_size))
      z = z + jnp.exp(0.5 * logsigma) * epsilon
    decode = Decoder(
      filters=self.decoder_start_filters,
      output_shape=self.image_shape,
      stage_length=len(self.encoder_stage_size) + 1,
    )(z, scales, train=train)
    logits = nn.Dense(self.class_num)(mu)  # add
    return distrib, decode, logits, scales

class TrainState(train_state.TrainState):
  batch_stats: dict
  sample_key: jax.random.KeyArray

from katacv.G_VAE.parser import VAEArgs
def get_learning_rate_fn(args: VAEArgs):
  if args.flag_cosine_schedule:
    return optax.cosine_decay_schedule(
      init_value=args.learning_rate,
      decay_steps=args.total_epochs * args.steps_per_epoch
    )
  return lambda x: x

def get_vae_model_state(args: VAEArgs, verbose=False):
  model = VAE(
    image_shape=args.image_shape,
    encoder_start_filters=args.encoder_start_filters,
    encoder_stage_size=args.encoder_stage_size,
    decoder_start_filters=args.decoder_start_filters,
    feature_size=args.feature_size
  )
  args.learning_rate_fn = get_learning_rate_fn(args)
  if verbose:
    print(model.tabulate(jax.random.PRNGKey(args.seed), jnp.empty(args.input_shape), train=False))
  variables = model.init(jax.random.PRNGKey(args.seed), jnp.empty(args.input_shape), train=False)
  return TrainState.create(
    apply_fn=model.apply,
    params=variables['params'],
    batch_stats=variables['batch_stats'],
    sample_key=jax.random.PRNGKey(args.seed),
    tx=optax.adam(learning_rate=args.learning_rate_fn if args.flag_cosine_schedule else args.learning_rate)
    # tx=optax.sgd(
    #   learning_rate=args.learning_rate_fn if args.flag_cosine_schedule else args.learning_rate,
    #   momentum=0.9, nesterov=True
    # )
  )

def get_decoder_state(args: VAEArgs, verbose=False):
  model = Decoder(
    filters=args.decoder_start_filters,
    output_shape=args.image_shape,
    stage_length=len(args.encoder_stage_size) + 1,
  )
  feature = jnp.empty((args.batch_size, args.feature_size))
  if verbose:
    print(model.tabulate(jax.random.PRNGKey(args.seed), feature, train=False))
  # variables = model.init(jax.random.PRNGKey(args.seed), feature, train=False)
  return TrainState.create(
    apply_fn=model.apply,
    params={},
    batch_stats={},
    sample_key=None,
    tx=optax.adam(learning_rate=args.learning_rate)
  )

def get_g_vae_model_state(args: VAEArgs, verbose=False):
  model = G_VAE(
    image_shape=args.image_shape,
    class_num=args.class_num,
    encoder_start_filters=args.encoder_start_filters,
    encoder_stage_size=args.encoder_stage_size,
    decoder_start_filters=args.decoder_start_filters,
    feature_size=args.feature_size
  )
  args.learning_rate_fn = get_learning_rate_fn(args)
  if verbose:
    print(model.tabulate(jax.random.PRNGKey(args.seed), jnp.empty(args.input_shape), train=False))
  variables = model.init(jax.random.PRNGKey(args.seed), jnp.empty(args.input_shape), train=False)
  return TrainState.create(
    apply_fn=model.apply,
    params=variables['params'],
    batch_stats=variables['batch_stats'],
    sample_key=jax.random.PRNGKey(args.seed),
    tx=optax.adam(learning_rate=args.learning_rate_fn if args.flag_cosine_schedule else args.learning_rate)
    # tx=optax.sgd(
    #   learning_rate=args.learning_rate_fn if args.flag_cosine_schedule else args.learning_rate,
    #   momentum=0.9, nesterov=True
    # )
  )

def test_vae():
  from katacv.G_VAE.parser import get_args_and_writer
  # args = get_args_and_writer(no_writer=True, model_name='VAE', dataset_name='cifar10')
  args = get_args_and_writer(no_writer=True, model_name='VAE', dataset_name='celeba', use_unet=True)
  state = get_vae_model_state(args, verbose=True)
  (distrib, logits), updates = state.apply_fn(
    {'params': state.params, 'batch_stats': state.batch_stats},
    x=jnp.empty(args.input_shape),
    train=True,
    rngs={'sample_key': state.sample_key},
    mutable=['batch_stats']
  )
  print(distrib[0].shape, distrib[1].shape, logits.shape)

def test_g_vae():
  from katacv.G_VAE.parser import get_args_and_writer
  # args = get_args_and_writer(no_writer=True, model_name='G-VAE', dataset_name='cifar10')
  args = get_args_and_writer(no_writer=True, model_name='G-VAE', dataset_name='celeba', use_unet=True)
  state = get_g_vae_model_state(args, verbose=True)
  (distrib, image, logits), updates = state.apply_fn(
    {'params': state.params, 'batch_stats': state.batch_stats},
    x=jnp.empty(args.input_shape),
    train=True,
    rngs={'sample_key': state.sample_key},
    mutable=['batch_stats']
  )
  print(distrib[0].shape, distrib[1].shape, image.shape, logits.shape)
  print(state.params.keys())
  print(jax.tree_map(lambda x: x.shape, state.params['Dense_0']))

if __name__ == '__main__':
  # test_classify_mnist()
  # test_vae()
  test_g_vae()
