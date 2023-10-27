import sys, os
sys.path.append(os.getcwd())

def convert_bytes(size):
    """ Convert bytes to KB, or MB or GB"""
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0 or x == 'TB':
            return "%3.1f %s" % (size, x)
        size /= 1024.0

from katacv.utils.related_pkgs.utility import *
from katacv.utils.related_pkgs.jax_flax_optax_orbax import *
from katacv.G_VAE.model import get_g_vae_model_state, get_vae_model_state  # change
from katacv.G_VAE.parser import get_args_and_writer

def parse_args():
    from katacv.utils.parser import argparse, cvt2Path
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=cvt2Path, default="/home/yy/Coding/models/G-VAE/celeba/G-VAE2048-0010",  # change
    # parser.add_argument("--path", type=cvt2Path, default="/home/yy/Coding/models/G-VAE/celeba/VAE2048-0010",  # change
        help="the original model weights path")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print(f"Loading the state from '{str(args.path)}'...")
    name = args.path.parent.parent.name
    vae_args = get_args_and_writer(no_writer=True, input_args=[], model_name=name, dataset_name="celeba")
    if 'G-VAE' in args.path.name:
        state = get_g_vae_model_state(vae_args)
    else:
        state = get_vae_model_state(vae_args)
    full_path = args.path
    with open(full_path, 'rb') as file:
        state = flax.serialization.from_bytes(state, file.read())
    print(f"Load weights from '{str(full_path)}' successfully!")

    # remove opt_state
    state = state.replace(opt_state=None)
    ckpter = ocp.PyTreeCheckpointer()

    save_args = orbax_utils.save_args_from_target(state)
    save_path = full_path.parent.joinpath(full_path.name + '-lite')
    ckpter.save(str(save_path), state, save_args=save_args, force=True)
    print(f"Save lite weight at '{str(save_path)}',")
    print(f"{convert_bytes(full_path.stat().st_size)}=>{convert_bytes(save_path.joinpath('checkpoint').stat().st_size)}")