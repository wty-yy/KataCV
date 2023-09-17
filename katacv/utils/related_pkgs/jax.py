"""
Import jax usefull packages conveniently.

from katacv.utils.related_pkgs.jax import *  # jax, jnp, flax, nn, train_state, optax
"""

import jax, jax.numpy as jnp
import flax, flax.linen as nn
from flax.training import train_state
import optax

from functools import partial