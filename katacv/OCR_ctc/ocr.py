from typing import Any
import jax, jax.numpy as jnp
import flax, optax, flax.linen as nn
from flax.training import train_state
import katacr.number_ocr.constant as const
from typing import Sequence, Callable
from functools import partial

def construct_B(y, L=const.total_length):
    B = jnp.concatenate([(jnp.eye(L) + jnp.eye(L, k=-1))[None, ...] for _ in range(y.shape[0])], 0)
    tmp = jnp.concatenate([jnp.eye(L, k=-2)[None, ...] for _ in range(y.shape[0])], 0)
    flag = (y != jnp.concatenate([y[:,:2], y[:,:-2]], -1))[:, None, :]
    B = B + flag * tmp
    return B

def get_label_length(y):
    tmp = jnp.where(y != 0, jnp.arange(y.shape[1]), 0)
    return jnp.argmax(tmp, -1) // 2 + 1

def mish(x):
    return x * jnp.tanh(jax.nn.softplus(x))

class ConvBlock(nn.Module):
    filters: int
    norm: nn.Module
    act: Callable
    kernel: Sequence[int] = (1, 1)
    strides: Sequence[int] = (1, 1)
    padding: str | Sequence[Sequence[int]] = 'SAME'
    use_norm: bool = True
    use_act: bool = True

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.filters, self.kernel, self.strides, self.padding, use_bias=not self.use_norm)(x)
        if self.use_norm: x = self.norm()(x)
        if self.use_act: x = self.act(x)
        return x

class ResBlock(nn.Module):
    conv: nn.Module
    act: Callable

    @nn.compact
    def __call__(self, x):
        residue = x
        n = x.shape[-1] // 2
        x = self.conv(filters=n, kernel=(1,1))(x)
        x = self.conv(filters=2*n, kernel=(3,3), use_act=False)(x)
        return residue + x

class Model(nn.Module):
    encoder_num: int
    stage_size = [1, 1, 2, 2]
    act: Callable = mish

    @nn.compact
    def __call__(self, x, train:bool=True):
        norm = partial(nn.BatchNorm, use_running_average=not train)
        conv = partial(ConvBlock, norm=norm, act=self.act)
        block = partial(ResBlock, conv=conv, act=self.act)
        x = conv(filters=32, kernel=(3,3))(x)
        for i, block_num in enumerate(self.stage_size):
            strides = (2, 2) if i < 2 else (2, 1)
            x = conv(filters=x.shape[-1]*2, kernel=(3,3), strides=strides)(x)
            for _ in range(block_num):
                x = block()(x)
        x = conv(filters=x.shape[-1], kernel=(2,2), padding=((0,0),(0,0)))(x)  # (N,1,15,512)
        x = conv(filters=self.encoder_num, kernel=(1,1), use_norm=False, use_act=False)(x)
        return x[:,0,...]  # (N,15,encoder_num)

if __name__ == '__main__':
    model = Model(output_size=const.encoder_num)
    print(model.tabulate(jax.random.PRNGKey(42), jnp.empty((1,32,64,1))))