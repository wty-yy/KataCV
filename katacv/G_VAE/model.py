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
  use_norm: bool = True
  use_act: bool = True

  @nn.compact
  def __call__(self, x):
    x = nn.ConvTranspose(self.filters, self.kernel, self.strides, use_bias=not self.use_norm)(x)
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

def mish(x):
  return x * jnp.tanh(jax.nn.softplus(x))

class Encoder(nn.Module):
  output_size: int
  stage_size: Sequence[int]
  act: Callable = mish

  @nn.compact
  def __call__(self, x, train: bool = True):
    norm = partial(nn.BatchNorm, use_running_average=not train)
    conv = partial(ConvBlock, norm=norm, act=self.act)
    block = partial(ResBlock, conv=conv, norm=norm, act=self.act)
    x = conv(filters=32, kernel=(3,3))(x)
    for block_num in self.stage_size:
      x = conv(filters=x.shape[-1]*2, kernel=(3,3), strides=(2,2))(x)
      for _ in range(block_num):
        x = block()(x)
    x = x.mean((1, 2))
    x = nn.Dense(self.output_size)(x)
    return x

class Decoder(nn.Module):
  output_shape: int
  stage_size: Sequence[int]
  act: Callable = mish

  @nn.compact
  def __call__(self, x, train: bool = True):
    norm = partial(nn.BatchNorm, use_running_average=not train)
    conv = partial(ConvBlock, norm=norm, act=self.act)
    conv_t = partial(ConvTransposeBlock, norm=norm, act=self.act)
    block = partial(ResBlock, conv=conv, norm=norm, act=self.act)
    power = 2 ** (len(self.stage_size))
    input_shape =(self.output_shape[0] // power, self.output_shape[1] // power, 32 * power)
    x = nn.Dense(input_shape[0] * input_shape[1] * input_shape[2])(x)
    x = x.reshape(x.shape[0], *input_shape)
    for block_num in self.stage_size:
      for _ in range(block_num):
        x = block()(x)
      x = conv_t(filters=x.shape[-1]//2, kernel=(3,3), strides=(2,2))(x)
      # x = jax.image.resize(x, (x.shape[0], x.shape[1]*2, x.shape[2]*2, x.shape[3]), method='nearest')
    x = conv(filters=self.output_shape[-1], kernel=(3,3))(x)
    return x

class VAE(nn.Module):
  image_shape: Sequence[int]
  encoder_stage_size: Sequence[int]
  decoder_stage_size: Sequence[int]
  feature_size: int = 128

  @nn.compact
  def __call__(self, x, train: bool = True):
    encode = Encoder(self.feature_size * 2, self.encoder_stage_size)(x, train=train)
    mu, logsigma = encode[:, :self.feature_size], encode[:, self.feature_size:]
    distrib = (mu, logsigma)
    z = mu
    if train:
      key = self.make_rng('sample_key')
      epsilon = jax.random.normal(key, (x.shape[0], self.feature_size))
      z = z + jnp.exp(0.5 * logsigma) * epsilon
    decode = Decoder(self.image_shape, self.decoder_stage_size)(z, train=train)
    return distrib, decode

class G_VAE(nn.Module):
  image_shape: Sequence[int]
  class_num: int
  encoder_stage_size: Sequence[int]
  decoder_stage_size: Sequence[int]
  feature_size: int = 128

  @nn.compact
  def __call__(self, x, train: bool = True):
    encode = Encoder(self.feature_size * 2, self.encoder_stage_size)(x, train=train)
    mu, logsigma = encode[:, :self.feature_size], encode[:, self.feature_size:]
    distrib = (mu, logsigma)
    z = mu
    if train:
      key = self.make_rng('sample_key')
      epsilon = jax.random.normal(key, (x.shape[0], self.feature_size))
      z = z + jnp.exp(0.5 * logsigma) * epsilon
    decode = Decoder(self.image_shape, self.decoder_stage_size)(z, train=train)
    logits = nn.Dense(self.class_num)(mu)  # add
    return distrib, decode, logits

class TrainState(train_state.TrainState):
  batch_stats: dict
  sample_key: jax.random.KeyArray

@partial(jax.jit, static_argnames='train')
def model_step(state: TrainState, x, y, train: bool = True):
  def loss_fn(params):
    logits, updates = state.apply_fn(
      {'params': params, 'batch_stats': state.batch_stats},
      x, train,
      mutable=['batch_stats']
    )
    loss = -jax.nn.log_softmax(logits)[jnp.arange(y.shape[0]), y].mean()
    return loss, (updates, logits)
  if train:
    (loss, (updates, logits)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.replace(batch_stats=updates['batch_stats'])
    state = state.apply_gradients(grads=grads)
  else:
    loss, (_, logits) = loss_fn(state.params)
  acc = (jnp.argmax(logits, -1) == y).mean()
  return state, loss, acc

def test_classify_mnist():
  model = Encoder(output_size=10, stage_size=(2,4))
  key = jax.random.PRNGKey(42)
  mnist_x = jnp.empty((32, 28, 28, 1))
  print(model.tabulate(key, mnist_x, train=False))
  variables = model.init(key, mnist_x, train=False)
  logits = model.apply(variables, mnist_x, train=False)
  print("test output shape:", logits.shape)

  # Training test on mnist classification
  parser = argparse.ArgumentParser()
  parser.add_argument("--total-epochs", type=int, default=3)
  parser.add_argument("--path-dataset", type=cvt2Path, default=Path("/home/yy/Coding/datasets/mnist"))
  parser.add_argument("--batch-size", type=int, default=64)
  parser.add_argument("--shuffle-size", type=int, default=64*16)
  args = parser.parse_args()

  state = TrainState.create(
    apply_fn=model.apply,
    params=variables['params'],
    batch_stats=variables['batch_stats'],
    sample_key=None,
    tx=optax.adam(learning_rate=1e-4)
  )

  from katacv.utils.mini_data.mnist import load_mnist
  from katacv.utils.mini_data.build_dataset import DatasetBuilder
  data = load_mnist(str(args.path_dataset))
  ds_builder = DatasetBuilder(data, args)
  ds_train, ds_train_size = ds_builder.get_dataset(subset='train')
  ds_val, ds_val_size = ds_builder.get_dataset(subset='val')
  for epoch in range(args.total_epochs):
    print(f"epoch: {epoch+1}/{args.total_epochs}:")
    print("training...")
    mean_loss, mean_acc = 0, 0
    bar = tqdm(ds_train, total=ds_train_size)
    for i, (x, y) in enumerate(bar):
      x, y = x.numpy(), y.numpy()
      state, loss, acc = model_step(state, x, y, train=True)
      mean_loss += (loss - mean_loss) / (i+1)
      mean_acc += (acc - mean_acc) / (i+1)
      bar.set_description(f"loss: {mean_loss:.5f}, acc: {mean_acc:.5f}")

    print("evaluating...")
    mean_loss, mean_acc = 0, 0
    bar = tqdm(ds_val, total=ds_val_size)
    for i, (x, y) in enumerate(bar):
      x, y = x.numpy(), y.numpy()
      _, loss, acc = model_step(state, x, y, train=False)
      mean_loss += (loss - mean_loss) / (i+1)
      mean_acc += (acc - mean_acc) / (i+1)
      bar.set_description(f"loss: {mean_loss:.5f}, acc: {mean_acc:.5f}")

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
    encoder_stage_size=args.encoder_stage_size,
    decoder_stage_size=args.decoder_stage_size,
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
  )

def get_decoder_state(args: VAEArgs, verbose=False):
  model = Decoder(
    output_shape=args.image_shape,
    stage_size=args.decoder_stage_size
  )
  feature = jnp.empty((args.batch_size, args.feature_size))
  if verbose:
    print(model.tabulate(jax.random.PRNGKey(args.seed), feature, train=False))
  variables = model.init(jax.random.PRNGKey(args.seed), feature, train=False)
  return TrainState.create(
    apply_fn=model.apply,
    params=variables['params'],
    batch_stats=variables['batch_stats'],
    sample_key=None,
    tx=optax.adam(learning_rate=args.learning_rate)
  )

def get_g_vae_model_state(args: VAEArgs, verbose=False):
  model = G_VAE(
    image_shape=args.image_shape,
    class_num=args.class_num,
    encoder_stage_size=args.encoder_stage_size,
    decoder_stage_size=args.decoder_stage_size,
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
  )

def test_vae():
  from katacv.G_VAE.parser import get_args_and_writer
  args = get_args_and_writer(no_writer=True)
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
  args = get_args_and_writer(no_writer=True, model_name='G-VAE', dataset_name='cifar10')
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
