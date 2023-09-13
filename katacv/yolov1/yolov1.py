import jax, jax.numpy as jnp
import flax, flax.linen as nn
from flax.training import train_state
import optax
from katacv.utils.imagenet.build_dataset import ImagenetBuilder
from yolov1_pretrain import Darknet
from katacv.utils.logs import Logs, MeanMetric
from pathlib import Path

logs = Logs(
    init_logs={
        'loss_train': MeanMetric(),
        'mAP_train': MeanMetric(),
        'loss_val': MeanMetric(),
        'mAP_val': MeanMetric(),
        'epoch': 0,
        'SPS': MeanMetric(),
        'SPS_avg': MeanMetric()
    },
    folder2name={
        'train/metrics': ['loss_train', 'mAP_train'],
        'val/metrics': ['loss_val', 'mAP_val'],
        'charts': ['SPS', 'SPS_avg', 'epoch']
    }
)

def get_args_and_writer():
    from katacv.utils.parser import Parser, cvt2Path
    parser = Parser()
    # VOC Dataset config
    parser.add_argument("--path-dataset-tfrecord", type=cvt2Path, default=Path("/home/wty/Coding/datasets/VOC/tfrecord"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--shuffle-size", type=int, default=128*16)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--split-size", type=int, default=7)
    args, writer = parser.get_args_and_writer()
    args.input_shape = (args.batch_size, args.image_size, args.image_size, 3)
    return args, writer

class YoloV1()