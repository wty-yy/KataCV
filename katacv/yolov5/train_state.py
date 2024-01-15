from katacv.utils.related_pkgs.jax_flax_optax_orbax import *
from flax import struct

class TrainState(train_state.TrainState):
  batch_stats: dict
  grads: dict
  accumulate: int
  acc_count: int
  tx_bias: optax.GradientTransformation = struct.field(pytree_node=False)

def apply_gradients(state: TrainState, grads: jnp.ndarray):
  def split(x, keep_bias=False):
    def fn(key, a):
      if not hasattr(key[-1], 'key'): return a
      if 'bias' in key[-1].key:
        if keep_bias: return a
        return jnp.zeros_like(a)
      elif keep_bias: return jnp.zeros_like(a)
      return a
    return jax.tree_util.tree_map_with_path(fn, x)
  opt_state = state.opt_state
  params = state.params
  updates1, opt_state1 = state.tx.update(split(grads, keep_bias=False), split(opt_state, keep_bias=False), params)
  updates2, opt_state2 = state.tx_bias.update(split(grads, keep_bias=True), split(opt_state, keep_bias=True), params)
  def merge(x, y):
    return jax.tree_map(lambda a, b: a + b, x, y)
  params = optax.apply_updates(params, merge(updates1, updates2))
  opt_state = (merge(opt_state1[0], opt_state2[0]), *opt_state1[1:])
  return state.replace(step=state.step+1, params=params, opt_state=opt_state)

def zeros_grads(state: TrainState):
  state = state.replace(
    grads=jax.tree_map(lambda x: jnp.zeros_like(x), state.grads)
  )
  return state

def update_grads(state: TrainState):
  # state = state.apply_gradients(grads=state.grads)
  state = apply_gradients(state, state.grads)
  state = zeros_grads(state)
  return state

def accumulate_grads(state: TrainState, grads: dict):
  state = state.replace(
    grads=jax.tree_map(lambda x, y: x + y, state.grads, grads),
    acc_count=state.acc_count+1
  )
  state = jax.lax.cond(
    state.acc_count % state.accumulate == 0,
    update_grads,
    lambda state: state,
    state
  )
  return state
