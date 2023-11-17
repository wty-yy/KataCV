from typing import Callable, Any, Tuple, Sequence, Optional, Union, List
from pathlib import Path
from tqdm import tqdm
import argparse, time
import math
def cvt2Path(x): return Path(x)
def str2bool(x): return x in ['yes', 'y', 'True', '1']
