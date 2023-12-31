import sys, os
sys.path.append(os.getcwd())

from typing import Callable, Any
import jax, jax.numpy as jnp
import flax, flax.linen as nn
from flax.training import train_state
import optax

ModuleDef = Any

class ConvBlock(nn.Module):
    features: int
    kernel_size: tuple[int]
    activation: Callable
    norm: ModuleDef
    strides: tuple[int] = (1, 1)
    padding: str | tuple = 'SAME'

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.features, self.kernel_size, self.strides, self.padding)(x)
        x = self.norm()(x)
        x = self.activation(x)
        return x

from functools import partial
class Darknet(nn.Module):
    conv: ModuleDef = ConvBlock
    max_pool: Callable = lambda x: nn.max_pool(x, (2, 2), strides=(2, 2))
    activation: Callable = lambda x: nn.leaky_relu(x, negative_slope=0.1)

    @nn.compact
    def __call__(self, x, train: bool = True):  # (N,224,224,3)
        norm = partial(nn.BatchNorm, use_running_average=not train)
        conv = partial(self.conv, activation=self.activation, norm=norm)
        x = conv(features=64, kernel_size=(7,7), strides=(2,2), padding=((3,3),(3,3)))(x)
        x = self.max_pool(x)
        x = conv(features=192, kernel_size=(3,3))(x)
        x = self.max_pool(x)

        x = conv(features=128, kernel_size=(1,1))(x)
        x = conv(features=256, kernel_size=(3,3))(x)
        x = conv(features=256, kernel_size=(1,1))(x)
        x = conv(features=512, kernel_size=(3,3))(x)
        x = self.max_pool(x)

        for _ in range(4):
            x = conv(features=256, kernel_size=(1,1))(x)
            x = conv(features=512, kernel_size=(3,3))(x)
        x = conv(features=512, kernel_size=(1,1))(x)
        x = conv(features=1024, kernel_size=(3,3))(x)
        x = self.max_pool(x)
        
        for _ in range(2):
            x = conv(features=512, kernel_size=(1,1))(x)
            x = conv(features=1024, kernel_size=(3,3))(x)
        
        return x

class YOLOv1PreModel(nn.Module):
    @nn.compact
    def __call__(self, x, train: bool = True):
        x = Darknet()(x, train)  # not need /255, since we use batch normalize
        x = nn.avg_pool(x, (7,7))
        x = x.reshape(x.shape[0], -1)
        x = nn.Dropout(0.4, deterministic=not train)(x)
        x = nn.Dense(1000)(x)
        return x

from katacv.utils.logs import Logs, MeanMetric

logs = Logs(
    init_logs={
        'loss_train': MeanMetric(),
        'accuracy_top1_train': MeanMetric(),
        'accuracy_top5_train': MeanMetric(),
        'loss_val': MeanMetric(),
        'accuracy_top1_val': MeanMetric(),
        'accuracy_top5_val': MeanMetric(),
        'epoch': 0,
        'SPS': MeanMetric(),
        'SPS_avg': MeanMetric()
    },
    folder2name={
        'train/metrics': ['loss_train', 'accuracy_top1_train', 'accuracy_top5_train'],
        'val/metrics': ['loss_val', 'accuracy_top1_val', 'accuracy_top5_val'],
        'charts': ['SPS', 'SPS_avg', 'epoch']
    }
)

def get_args_and_writer():
    from katacv.utils.parser import Parser, Path, cvt2Path
    parser = Parser(model_name="YOLOv1PreTrain", wandb_project_name="Imagenet2012")
    # Imagenet dataset
    parser.add_argument("--path-dataset-tfrecord", type=cvt2Path, default=Path("/media/yy/Data/dataset/imagenet/tfrecord"),
        help="the path of the tfrecord dataset directory")
    parser.add_argument("--image-size", type=int, default=224,
        help="the input image size of the model")
    parser.add_argument("--image-center-crop-padding-size", type=int, default=32,
        help="the padding size of the center crop of the origin image")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the size of each batch")
    parser.add_argument("--shuffle-size", type=int, default=256*16,
        help="the shuffle size of the dataset")
    parser.add_argument("--total-epochs", type=int, default=40,
        help="the total epochs of the training")
    parser.add_argument("--learning-rate", type=float, default=1e-3,
        help="the learning rate of the optimizer")
    
    args, writer = parser.get_args_and_writer()
    args.input_shape = (args.batch_size, args.image_size, args.image_size, 3)
    return args, writer

class TrainState(train_state.TrainState):
    batch_stats: dict
    dropout_key: jax.random.KeyArray

from functools import partial
@partial(jax.jit, static_argnames='train')
def model_step(state: TrainState, x, y, train: bool = True):

    def loss_fn(params):
        logits, updates = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            x, train=train,
            mutable=['batch_stats'],
            rngs={'dropout': jax.random.fold_in(state.dropout_key, state.step)}
        )
        loss = -(y * jax.nn.log_softmax(logits)).sum(-1).mean()
        return loss, (logits, updates)
    
    if train:
        (loss, (logits, updates)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        state = state.replace(batch_stats=updates['batch_stats'])
    else:
        loss, (logits, _) = loss_fn(state.params)

    y_cat = jnp.argmax(y, -1)
    top1 = (jnp.argmax(logits, -1) == y_cat).mean()
    top5_pred = jnp.argsort(logits, axis=-1)[:, -5:]
    top5 = jnp.any(top5_pred == y_cat[:, jnp.newaxis], axis=-1).mean()

    return state, loss, top1, top5

def get_pretrain_state(args=None, verbose=False):
    seed, input_shape, learning_rate = (0, (1,224,224,3), 1e-3) if args is None else (args.seed, args.input_shape, args.learning_rate)
    model = YOLOv1PreModel()
    key = jax.random.PRNGKey(seed)
    if verbose:
        print(model.tabulate(key, jnp.empty(input_shape), train=False))

    variables = model.init(key, jnp.empty(input_shape), train=False)
    return TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optax.adam(learning_rate=learning_rate),
        batch_stats=variables['batch_stats'],
        dropout_key=key
    )

if __name__ == '__main__':
    args, writer = get_args_and_writer()
    state = get_pretrain_state(args, verbose=True)
    save_id = args.load_id + 1
    if save_id > 1:  # load_id > 0
        load_path = args.path_cp.joinpath(f"{args.model_name}-{save_id-1:04}")
        assert(load_path.exists())
        with open(load_path, 'rb') as file:
            state = flax.serialization.from_bytes(state, file.read())
        print(f"Successfully load weights from '{str(load_path)}'")
    
    save_path = args.path_cp.joinpath(f"{args.model_name}-{save_id:04}")
    if save_path.exists():
        print(f"The weights file '{str(save_path)}' is exists, still want to continue? [enter]", end="")
        input()
    
    from katacv.utils.imagenet.build_dataset import ImagenetBuilder
    ds_builder = ImagenetBuilder(args)
    train_ds, train_ds_size = ds_builder.get_dataset(sub_dataset='train')
    val_ds, val_ds_size = ds_builder.get_dataset(sub_dataset='val')

    import time
    from tqdm import tqdm

    start_time, global_step = time.time(), 0
    if args.train:
        for epoch in range(1,args.total_epochs+1):
            print(f"epoch: {epoch}/{args.total_epochs}")
            logs.reset()
            print("training...")
            for x, y in tqdm(train_ds, total=train_ds_size, desc="Processing"):
            # for x, y in tqdm(train_ds.take(300), total=300, desc="Processing"):
                global_step += 1
                state, *metrics = model_step(state, x.numpy(), y.numpy())
                logs.update(
                    ['loss_train', 'accuracy_top1_train', 'accuracy_top5_train'],
                    metrics
                )
                if global_step % args.write_tensorboard_freq == 0:
                    logs.update(
                        ['SPS', 'SPS_avg', 'epoch'],
                        [args.write_tensorboard_freq/logs.get_time_length(), global_step/(time.time()-start_time), epoch]
                    )
                    logs.start_time = time.time()
                    logs.writer_tensorboard(writer, global_step)

            logs.reset()
            print("validating...")
            for x, y in tqdm(val_ds, total=val_ds_size, desc="Processing"):
                state, *metrics = model_step(state, x.numpy(), y.numpy(), train=False)
                logs.update(
                    ['loss_val', 'accuracy_top1_val', 'accuracy_top5_val', 'epoch'],
                    metrics + [epoch]
                )
            logs.writer_tensorboard(writer, global_step)

            if epoch % args.save_weights_freq == 0:
                path = args.path_cp.joinpath(f"{args.model_name}-{save_id:04}")
                with open(path, 'wb') as file:
                    file.write(flax.serialization.to_bytes(state))
                print(f"save weights at '{str(path)}'")
                save_id += 1
    elif args.evaluate:
        print("evaluate on train dataset:")
        for x, y in tqdm(train_ds, total=train_ds_size, desc="Processing"):
            global_step += 1
            state, *metrics = model_step(state, x.numpy(), y.numpy(), train=False)
            logs.update(
                ['loss_train', 'accuracy_top1_train', 'accuracy_top5_train'],
                metrics
            )
            if global_step % args.write_tensorboard_freq == 0:
                logs.update(
                    ['SPS', 'SPS_avg'],
                    [args.write_tensorboard_freq/logs.get_time_length(), global_step/(time.time()-start_time)]
                )
                logs.start_time = time.time()
                logs.writer_tensorboard(writer, global_step)
        print("evaluate on val dataset:")
        for x, y in tqdm(val_ds, total=val_ds_size, desc="Processing"):
            state, *metrics = model_step(state, x.numpy(), y.numpy(), train=False)
            logs.update(
                ['loss_val', 'accuracy_top1_val', 'accuracy_top5_val'],
                metrics
            )
            logs.writer_tensorboard(writer, global_step)
        print(logs.to_dict())

    writer.close()