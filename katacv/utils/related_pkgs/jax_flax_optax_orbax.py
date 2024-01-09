"""
Import jax usefull packages conveniently.

from katacv.utils.related_pkgs.jax import *  # jax, jnp, flax, nn, train_state, optax
"""

import jax, jax.numpy as jnp
import flax, flax.linen as nn
from flax.training import train_state
from flax.training import orbax_utils
import optax
import orbax.checkpoint as ocp

from functools import partial

import os
# https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.97'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'  # allocate GPU memory as needed