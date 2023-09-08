import jax, jax.numpy as jnp
import flax, flax.linen as nn
import optax
from tensorboardX import SummaryWriter
import wandb
from argparse import ArgumentParser

def str2bool(x):
    return x in ['yes', 'y', 'True', '1']

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--wandb-track", type=str2bool, default=False, const=True, nargs='?',
        help="if taggled, track with wandb")
    parser.add_argument("--wandb-project-name", type=str, default="")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
        help="the learning rate of the adam")
    parser.add_argument("--pat-dataset", type=str, default=r"/media/yy/Data/dataset/imagenet",
        help="the path of the dataset")
    parser.add_argument("--load-weights", default=False, nargs=2,
        help="if load the weights, you should pass the name of weights in './logs/checkpoints/'")
    return parser.parse_args()

def build_model():
    

if __name__ == '__main__':
    args = parse_args()
    model = build_model()

