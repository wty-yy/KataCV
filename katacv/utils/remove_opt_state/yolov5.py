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
from katacv.yolov5.model import get_state  # change
from katacv.yolov5.parser import get_args_and_writer

def parse_args():
    from katacv.utils.parser import argparse, cvt2Path
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=cvt2Path, default="/home/yy/Coding/models/YOLOv5/YOLOv5_b32_scratch_stopD-0300",  # change
        help="the original model weights path")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    print(f"Loading the state from '{str(args.path)}'...")
    state = get_state(get_args_and_writer(no_writer=True, input_args=""), use_init=False)
    full_path = args.path
    with open(full_path, 'rb') as file:
        state = flax.serialization.from_bytes(state, file.read())
    print(f"Load weights from '{str(full_path)}' successfully!")

    # remove opt_state
    state = state.replace(opt_state=None, grads=None)
    ckpter = ocp.PyTreeCheckpointer()

    save_args = orbax_utils.save_args_from_target(state)
    save_path = full_path.parent / (full_path.name + '-lite')
    ckpter.save(str(save_path), state, save_args=save_args, force=True)
    print(f"Save lite weight at '{str(save_path)}',")
    print(f"{convert_bytes(full_path.stat().st_size)}=>{convert_bytes(save_path.joinpath('checkpoint').stat().st_size)}")