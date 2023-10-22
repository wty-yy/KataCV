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

if __name__ == '__main__':
  model = Encoder(output_size=10, stage_size=(2,4))
  key = jax.random.PRNGKey(42)
  mnist_x = jnp.empty((1, 28, 28, 1))
  print(model.tabulate(key, mnist_x, train=False))
  variables = model.init(key, mnist_x, train=False)
  logits = model.apply(variables, mnist_x, train=False)
  print(logits.shape)
