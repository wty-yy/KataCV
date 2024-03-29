import jax, jax.numpy as jnp
import flax.linen as nn
import flax, optax
import numpy as np
from flax.training import train_state
from typing import Callable, Sequence

class TrainState(train_state.TrainState):
  dropout_rng: jax.Array

class MainCLS:
  def get_vars(self):
    ret = {}
    for name in dir(self):
      val = getattr(self, name)
      if not callable(val) and '__' not in name:
        ret[name] = val
    return ret

  def __repr__(self):
    return str(self.get_vars())

class GPTConfig(MainCLS):
  n_embd = 768
  n_head = 12
  n_block = 12
  p_drop_embd = 0.1
  p_drop_resid = 0.1
  p_drop_attn = 0.1

  def __init__(self, n_vocab, n_token, **kwargs):
    self.n_vocab, self.n_token = n_vocab, n_token
    for k, v in kwargs.items():
      setattr(self, k, v)
    assert self.n_embd % self.n_head == 0, "n_embd must be devided by n_head"

class TrainConfig(MainCLS):
  seed = 42
  weight_decay = 0.1
  lr = 3e-4
  total_epochs = 100
  batch_size = 128
  betas = (0.9, 0.95)  # Adamw beta1, beta2
  warmup_tokens = 128*128*256  # 375e6
  grad_norm_clip = 1.0
  lr_fn: Callable

  def __init__(self, steps_per_epoch, n_token, **kwargs):
    self.steps_per_epoch = steps_per_epoch
    self.n_token = n_token
    for k, v in kwargs.items():
      setattr(self, k, v)

class CausalSelfAttention(nn.Module):
  n_embd: int  # NOTE: n_embd % n_head == 0
  n_head: int
  p_drop_attn: float

  @nn.compact
  def __call__(self, x: jnp.ndarray, train: bool, mask_len: Sequence[int] = None):
    D = self.n_embd // self.n_head  # hidden dim
    B, L, _ = x.shape  # Bachsize, token length, embedding dim
    mask = jnp.expand_dims(jnp.tri(L), (0, 1))  # Only consider previous token values
    if mask_len is not None:
      mask = jnp.where(jnp.arange(L).reshape(1, 1, L, 1).repeat(B, 0) >= mask_len.reshape(B, 1, 1, 1), 0, mask)
    x = nn.Dense(3 * self.n_embd)(x)
    q, k, v = jnp.array_split(x.reshape(B, L, self.n_head, -1).transpose(0, 2, 1, 3), 3, -1)
    attn = q @ jnp.swapaxes(k, -1, -2) / jnp.sqrt(D)
    attn = jnp.where(mask == 0, -1e18, attn)
    attn = jax.nn.softmax(attn)
    attn = nn.Dropout(self.p_drop_attn)(attn, deterministic=not train)
    y = (attn @ v).transpose(0, 2, 1, 3).reshape(B, L, self.n_embd)
    y = nn.Dense(self.n_embd)(y)
    return y

class AttentionBlock(nn.Module):
  cfg: GPTConfig

  @nn.compact
  def __call__(self, x: jnp.ndarray, train: bool, mask_len: Sequence[int] = None):
    attn_cfg = {key: getattr(self.cfg, key) for key in ['n_embd', 'n_head', 'p_drop_attn']}
    z = nn.LayerNorm()(x)
    z = CausalSelfAttention(**attn_cfg)(z, train, mask_len)
    x = x + nn.Dropout(self.cfg.p_drop_resid)(z, deterministic=not train)
    z = nn.Sequential([
      nn.LayerNorm(),
      nn.Dense(4*self.cfg.n_embd), nn.gelu,
      nn.Dense(self.cfg.n_embd),
    ])(x)
    x = x + nn.Dropout(self.cfg.p_drop_resid)(z, deterministic=not train)
    return x

class GPT(nn.Module):  # For Text
  cfg: GPTConfig

  @nn.compact  # x: (B, L, Nv)
  def __call__(self, x: jnp.ndarray, train: bool, mask_len: Sequence[int] = None):
    cfg = self.cfg
    pos_embd = self.param('pos_embd', lambda _, shape: jnp.zeros(shape), (1, cfg.n_token, cfg.n_embd))
    x = pos_embd + nn.Embed(cfg.n_vocab, cfg.n_embd)(x)  # (B, L, Ne)
    x = nn.Dropout(cfg.p_drop_embd)(x, deterministic=not train)
    for _ in range(cfg.n_block):
      x = AttentionBlock(cfg)(x, train, mask_len)
    x = nn.LayerNorm()(x)
    x = nn.Dense(cfg.n_vocab)(x)
    return x
    
  def get_state(self, train_cfg: TrainConfig, verbose: bool = False, load_path: str = None, train: bool = True) -> TrainState:
    def check_decay_params(kp, x):
      fg = x.ndim > 1
      for k in kp:
        if k.key in ['pos_embd', 'LayerNorm', 'Embed']:
          fg = False; break
      return fg
    def lr_fn():
      warmup_steps = train_cfg.warmup_tokens // (train_cfg.n_token * train_cfg.batch_size)
      warmup_fn = optax.linear_schedule(0.0, train_cfg.lr, warmup_steps)
      second_steps = max(train_cfg.total_epochs * train_cfg.steps_per_epoch - warmup_steps, 1)
      second_fn = optax.cosine_decay_schedule(
        train_cfg.lr, second_steps, 0.1
      )
      return optax.join_schedules(
        schedules=[warmup_fn, second_fn],
        boundaries=[warmup_steps]
      )
    rng = jax.random.PRNGKey(train_cfg.seed)
    if not train:  # return state with apply function
      return TrainState.create(apply_fn=self.apply, params={'a': 1}, tx=optax.sgd(1), dropout_rng=rng)
    examp = jnp.empty((train_cfg.batch_size, self.cfg.n_token), jnp.int32)
    if verbose: print(self.tabulate(rng, examp, train=False))
    variables = self.init(rng, examp, train=False)
    print("mini-GPT params:", sum([np.prod(x.shape) for x in jax.tree_util.tree_flatten(variables)[0]]))
    decay_mask = jax.tree_util.tree_map_with_path(check_decay_params, variables['params'])
    train_cfg.lr_fn = lr_fn()
    state = TrainState.create(
      apply_fn=self.apply,
      params=variables['params'],
      # AdamW is Adam with weight decay
      tx=optax.chain(
        optax.clip_by_global_norm(train_cfg.grad_norm_clip),
        optax.adamw(train_cfg.lr_fn, train_cfg.betas[0], train_cfg.betas[1], weight_decay=train_cfg.weight_decay, mask=decay_mask),
      ),
      dropout_rng=rng,
    )
    if load_path is not None:
      with open(load_path, 'rb') as file:
        state = flax.serialization.from_bytes(state, file.read())
      print(f"Load weights from {load_path}")
    return state
  
  def create_fns(self):
    def model_step(state: TrainState, x: jnp.ndarray, y: jnp.ndarray, train: bool):
      dropout_rng, base_rng = jax.random.split(state.dropout_rng)
      def loss_fn(params):
        logits = state.apply_fn({'params': params}, x, train=train, rngs={'dropout': dropout_rng})
        tmp = -jax.nn.log_softmax(logits).reshape(-1, logits.shape[-1])
        loss = tmp[jnp.arange(tmp.shape[0]), y.reshape(-1)].mean()
        acc = (jnp.argmax(logits, -1).reshape(-1) == y.reshape(-1)).mean()
        return loss, acc
      (loss, acc), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
      state = state.apply_gradients(grads=grads)
      state = state.replace(dropout_rng=base_rng)
      return state, (loss, acc)
    self.model_step = jax.jit(model_step, static_argnames='train')

    def predict(state: TrainState, x: jnp.ndarray, rng: jax.Array, mask_len: Sequence[int] = None):
      logits = state.apply_fn({'params': state.params}, x, train=False, mask_len=mask_len)
      if mask_len is not None:
        logits = logits[jnp.arange(logits.shape[0]), mask_len-1, :]  # (B, n_vocab)
      pred = jax.random.categorical(rng, logits, -1)
      return pred
    self.predict = jax.jit(predict)

  def save_model(self, state, save_path):
    with open(save_path, 'wb') as file:
      file.write(flax.serialization.to_bytes(state))
    print(f"Save weights to {save_path}")
  
if __name__ == '__main__':
  batch_size = 128
  n_vocab = 1000
  n_len = 128  # 86,691,304
  # n_len = 90  # 86,662,120
  n_embd = 768
  n_head = 12
  n_block = 12
  gpt_cfg = GPTConfig(n_vocab, n_len, n_embd=n_embd, n_head=n_head, n_block=n_block)
  gpt = GPT(gpt_cfg)
  # rng = jax.random.PRNGKey(42)
  # x = jax.random.randint(rng, (batch_size, n_len), 0, 6)
  # print(gpt.tabulate(rng, x, train=False))
  # variable = gpt.init(rng, x, train=False)
  # print("params:", sum([np.prod(x.shape) for x in jax.tree_util.tree_flatten(variable)[0]]))
  train_cfg = TrainConfig(steps_per_epoch=512, n_token=128)
  state = gpt.get_state(train_cfg, verbose=True)
