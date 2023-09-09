import sys, datetime
from pathlib import Path
from typing import Any
sys.path.append(str(Path.cwd()))

import jax, jax.numpy as jnp
import flax, flax.linen as nn
from flax.training.train_state import TrainState
import optax
from tensorboardX import SummaryWriter
from argparse import ArgumentParser
from utils.logs import Logs, MeanMetric

logs = Logs(
    init_logs={
        'loss': MeanMetric(),
        'accuracy_top1': MeanMetric(),
        'accuracy_top5': MeanMetric(),
        'SPS': MeanMetric(),
        'SPS_avg': MeanMetric()
    }
    folder2name={
        'metrics': ['loss', 'top1_accuracy', 'top5_accuracy'],
        'charts': ['SPS', 'SPS_average']
    }
)

def get_args_and_writer():
    cvt2Path = lambda x: Path(x)
    str2bool = lambda x: x in ['yes', 'y', 'True', '1']
    parser = ArgumentParser()
    parser.add_argument("--model-name", type=str, default="VGG16",
        help="the name of the model")
    parser.add_argument("--wandb-track", type=str2bool, default=False, const=True, nargs='?',
        help="if taggled, track with wandb")
    parser.add_argument("--wandb-project-name", type=str, default="Imagenet2012")
    parser.add_argument("--path-logs", type=cvt2Path, default=Path.cwd().joinpath("logs"),
        help="the path of the logs")
    parser.add_argument("--load-weights-id", type=int, default=0,
        help="if load the weights, you should pass the id of weights in './logs/{model_name}-checkpoints/{model_name}-{id:04}'")
    parser.add_argument("--save_weights_freq", type=int, default=1,
        help="the frequency to save the weights in './logs/{model_name}-checkpoints/{model_name}-{id:04}'")
    # build dataset params
    parser.add_argument("--path-dataset-tfrecord", type=str, default=r"/media/yy/Data/dataset/imagenet",
        help="the path of the dataset")
    parser.add_argument("--image-size", type=int, default=224,
        help="the image size inputs to the model")
    parser.add_argument("--image-center-crop-padding-size", type=int, default=32,
        help="the padding size when crop the image by center")
    parser.add_argument("--batch-size", type=int, default=128,
        help="the batch size for training the model")
    parser.add_argument("--shuffle-size", type=int, default=128*16,
        help="the shuffle size of the dataset")
    # hyper-params
    parser.add_argument("--seed", type=int, default=1,
        help="the seed for initalizing the model")
    parser.add_argument("--total-epochs", type=int, default=20,
        help="the total epochs of the training")
    parser.add_argument("--learning-rate", type=float, default=1e-5,
        help="the learning rate of the adam")
    args = parser.parse_args()

    args.path_logs.mkdir(exist_ok=True)
    args.path_cp = args.path_logs.joinpath(args.model_name+"-checkpoints")
    args.path_cp.mkdir(exist_ok=True)

    args.input_shape = (args.batch_size, args.image_size, args.image_size, 3)
    args.run_name = f"{args.model_name}__{args.seed}__{datetime.datetime.now().strftime(r'%Y%m%d_%H%M%S')}".replace("/", "-")
    if args.wandb_track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            sync_tensorboard=True,
            config=vars(args),
            name=args.run_name,
            save_code=True,
        )
    writer = SummaryWriter(args.path_logs.joinpath(args.run_name))
    writer.add_text(
        "hyper-parameters",
        "|param|value|\n|-|-|\n%s" % ('\n'.join([f"|{key}|{value}|" for key, value in vars(args).items()]))
    )
    return args, writer

from typing import Sequence

class ConvBlock(nn.Module):
    cnn_size: int
    block_size: int

    @nn.compact
    def __call__(self, x):
        for _ in range(self.block_size):
            x = nn.relu(nn.Conv(self.cnn_size, (3, 3))(x))
        return nn.max_pool(x, (2, 2), (2, 2))

class VGG(nn.Module):
    cnn_sizes: Sequence[int]
    block_sizes: Sequence[int]

    @nn.compact
    def __call__(self, x):
        x = x / 255.0
        for cnn_size, block_size in zip(self.cnn_sizes, self.block_sizes):
            x = ConvBlock(cnn_size, block_size)(x)
        x = x.reshape(x.shape[0], -1)
        x = nn.relu(nn.Dense(4096)(x))
        x = nn.relu(nn.Dense(4096)(x))
        return nn.Dense(1000)(x)

if __name__ == '__main__':
    args, writer = get_args_and_writer()
    VGG16 = VGG(cnn_sizes=[64, 128, 256, 512, 512], block_sizes=[2, 2, 3, 3, 3])
    VGG19 = VGG(cnn_sizes=[64, 128, 256, 512, 512], block_sizes=[2, 2, 4, 4, 4])
    model = VGG16 if args.model_name == 'VGG16' else VGG19
    key = jax.random.PRNGKey(args.seed)
    print(model.tabulate(key, jnp.empty(args.input_shape)))

    state = TrainState.create(
        apply_fn=model.apply,
        params=model.init(key, jnp.empty(args.input_shape)),
        tx=optax.adam(learning_rate=args.learning_rate)
    )
    
    save_id = args.load_weights_id + 1
    if save_id > 1:  # load_weights_id > 0
        load_path = args.path_cp.joinpath(f"{args.model_name}-{save_id-1:04}")
        assert(load_path.exists())
        with open(load_path, 'rb') as file:
            state = flax.serialization.from_bytes(state, file.read())
        print(f"Successfully load weights from '{str(load_path)}'")
    
    save_path = args.path_cp.joinpath(f"{args.model_name}-{save_id:04}")
    if save_path.exists():
        print(f"The weights file '{str(save_path)}' is exists, still want to continue? [enter]", end="")
        input()
    
    from utils.build_dataset import DatasetBuilder
    ds_builder = DatasetBuilder(args)
    train_ds = ds_builder.get_dataset(train=True)
    val_ds = ds_builder.get_dataset(train=False)

    @jax.jit
    def train_step(state: TrainState, x, y):

        def loss_fn(params, x, y):
            logits = state.apply_fn(params, x)
            loss = -(y * jax.nn.log_softmax(logits)).sum(-1)
            return loss.mean(), logits
        
        (loss, logits), grads = jax.value_and_grad(loss_fn)(state.params, x, y)
        state = state.apply_gradients(grads)

        y_cat = jnp.argmax(y, -1)
        top1 = (jnp.argmax(logits, -1) == y_cat).mean()
        top5_pred = jnp.argsort(logits, axis=-1)[:, -5:]
        top5 = jnp.any(top5_pred == y_cat[:, jnp.newaxis], axis=-1).mean()

        return state, loss, top1, top5

    import time
    from tqdm import tqdm

    start_time, global_step = time.time(), 0
    for epoch in range(args.total_epochs):
        print(f"epoch: {epoch+1}/{args.total_epochs}")
        logs.reset()

        for x, y in tqdm(train_ds):
            logs.start_time = time.time()
            global_step += 1
            state, *metrics = train_step(state, x, y)
            logs.update(
                ['loss', 'accuracy_top1', 'accuracy_top5', 'SPS', 'SPS_avg'],
                metrics + [int(logs.get_time_length()), int(global_step/(time.time()-start_time))]
            )
            logs.writer_tensorboard(writer, global_step)

        if epoch % args.save_weights_freq == 0:
            path = args.path_cp.joinpath(f"{args.model_name}-{save_id:04}")
            with open(path, 'wb') as file:
                file.write(flax.serialization.to_bytes(state))
            print(f"save weights at '{str(path)}'")
            save_id += 1

    writer.close()
