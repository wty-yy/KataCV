import jax, jax.numpy as jnp
import flax, flax.linen as nn
from flax.training import train_state
import optax
from katacv.utils.imagenet.build_dataset import ImagenetBuilder
from yolov1_pretrain import Darknet, ConvBlock, partial
from katacv.utils.logs import Logs, MeanMetric
from pathlib import Path
from typing import Callable

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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shuffle-size", type=int, default=64*16)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--split-size", type=int, default=7)
    args, writer = parser.get_args_and_writer()
    args.input_shape = (args.batch_size, args.image_size, args.image_size, 3)
    return args, writer

class YoloV1(nn.Module):
    activation: Callable = lambda x: nn.leaky_relu(x, negative_slope=0.1)
    S: int = 7  # split_size
    B: int = 2  # bounding_box_num
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        x = Darknet(activation=self.activation)(x, train)
        norm = partial(nn.BatchNorm, use_running_average=not train)
        conv = partial(ConvBlock, activation=self.activation, norm=norm)
        x = conv(features=1024, kernel_size=(3,3))(x)
        x = conv(features=1024, kernel_size=(3,3), strides=(2,2), padding=((1,1),(1,1)))(x)
        x = conv(features=1024, kernel_size=(3,3))(x)
        x = conv(features=1024, kernel_size=(3,3))(x)
        x = x.reshape(x.shape[0], -1)
        x = self.activation(nn.Dense(1024)(x))
        x = nn.Dropout(0.5, deterministic=not train)(x)
        x = nn.Dense(self.S*self.S*(20+5*self.B))(x)
        return x

if __name__ == '__main__':
    args, writer = get_args_and_writer()
    key = jax.random.PRNGKey(args.seed)
    model = YoloV1()
    print(model.tabulate(key, jnp.empty(args.input_shape), train=False))
