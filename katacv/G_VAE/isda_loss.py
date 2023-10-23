from katacv.utils.related_pkgs.jax_flax_optax_orbax import *
from katacv.utils.related_pkgs.utility import *

@partial(jax.jit, static_argnames=['class_num', 'log_eps'])
def isda_loss(mu, logsigma2, w, b, label, class_num, log_eps=-1e5):
  B, K, C = mu.shape[0], mu.shape[1], b.shape[0]
  mu, logsigma2 = mu[..., None], logsigma2[..., None]
  w = jnp.repeat(w[None,...], B, axis=0)  # (B,K,C)
  b = jnp.repeat(b[None,...], B, axis=0)  # (B,C,)
  onehot = jax.nn.one_hot(label, class_num)
  w_y = jnp.einsum('bkc,bc->bk', w, onehot)[...,None]
  b_y = jnp.einsum('bc,bc->b', b, onehot)[...,None]
  def loop_func(pre, x):
    ret = jnp.logaddexp(pre, x)
    return ret, ret
  xs = jnp.transpose(
    0.5 * ((w - w_y) ** 2 * jnp.exp(logsigma2)).sum(1)
    + ((w - w_y) * mu).sum(1)
    + b - b_y, (1,0))  # (B,C) -> (C,B)
  init = jnp.zeros((B,)) + log_eps
  ret, _ = jax.lax.scan(loop_func, init, xs)  # (B,)
  loss = ret.mean()
  # direct_loss = jnp.log(
  #   jnp.exp(
  #     0.5 * ((w - w_y) ** 2 * jnp.exp(logsigma2)).sum(1)
  #     + ((w - w_y) * mu).sum(1)
  #     + b - b_y
  #   ).sum(-1)
  # ).mean()
  # print(direct_loss)
  return loss

if __name__ == '__main__':
  B = 32
  K = 128
  C = 10
  import numpy as np
  np.random.seed(42)
  mu = np.random.randn(B, K)
  logsigma2 = np.random.randn(B, K)
  w = np.random.randn(K, C)
  b = np.random.randn(C)
  y = np.random.randint(0, C, size=(32,))
  loss = isda_loss(mu, logsigma2, w, b, y, class_num=C)
  print(loss)
