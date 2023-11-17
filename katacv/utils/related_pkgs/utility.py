from typing import Callable, Any, Tuple, Sequence, Optional, Union, List
from pathlib import Path
from tqdm import tqdm
import argparse, time
import math
def cvt2Path(x): return Path(x)
def str2bool(x): return x in ['yes', 'y', 'True', '1']

import os
# https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.90'
