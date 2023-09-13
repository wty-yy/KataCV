# -*- coding: utf-8 -*-
'''
@File    : googlenet_jax.py
@Time    : 2023/09/10 21:34:18
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : 
2023/09/10: 完成googlenet(Inception-v1)框架
2023/09/11: 开始googlenet训练
2023/09/12: 训练到50个epochs
'''
import sys, datetime
from pathlib import Path
sys.path.append(str(Path.cwd()))

import jax, jax.numpy as jnp
import flax, flax.linen as nn
from flax.training import train_state
import optax
from tensorboardX import SummaryWriter
from argparse import ArgumentParser
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
    cvt2Path = lambda x: Path(x)
    str2bool = lambda x: x in ['yes', 'y', 'True', '1']
    parser = ArgumentParser()
    parser.add_argument("--model-name", type=str, default="GoogleNet",
        help="the name of the model")
    parser.add_argument("--wandb-track", type=str2bool, default=False, const=True, nargs='?',
        help="if taggled, track with wandb")
    parser.add_argument("--wandb-project-name", type=str, default="Imagenet2012")
    parser.add_argument("--path-logs", type=cvt2Path, default=Path.cwd().joinpath("logs"),
        help="the path of the logs")
    parser.add_argument("--load-weights-id", type=int, default=0,
        help="if load the weights, you should pass the id of weights in './logs/{model_name}-checkpoints/{model_name}-{id:04}'")
    parser.add_argument("--save-weights-freq", type=int, default=1,
        help="the frequency to save the weights in './logs/{model_name}-checkpoints/{model_name}-{id:04}'")
    parser.add_argument("--val-sample-size", type=int, default=50000,
        help="the size of the val-dataset to validate after each training epoch")
    parser.add_argument("--write-tensorboard-freq", type=int, default=100,
        help="the frequeny of writing the tensorboard")
    # build dataset params
    parser.add_argument("--path-dataset-tfrecord", type=str, default="/media/yy/Data/dataset/imagenet/tfrecord/",
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
    parser.add_argument("--learning-rate", type=float, default=1e-3,
        help="the learning rate of the adam")
    args = parser.parse_args()

    args.path_logs.mkdir(exist_ok=True)
    args.path_cp = args.path_logs.joinpath(args.model_name+"-checkpoints")
    args.path_cp.mkdir(exist_ok=True)

    args.val_sample_batch = args.val_sample_size // args.batch_size
    args.input_shape = (args.batch_size, args.image_size, args.image_size, 3)
    args.run_name = f"{args.model_name}__load_{args.load_weights_id}__lr_{args.learning_rate}__{datetime.datetime.now().strftime(r'%Y%m%d_%H%M%S')}".replace("/", "-")
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

class Inception(nn.Module):
    conv1x1: int
    conv3x3: tuple[int]
    conv5x5: tuple[int]
    pool1x1: int

    @nn.compact
    def __call__(self, x):  # nn.Conv default padding is 'SAME'
        path1 = nn.relu(nn.Conv(self.conv1x1, (1, 1))(x))
        path2 = nn.relu(nn.Conv(self.conv3x3[0], (1, 1))(x))
        path2 = nn.relu(nn.Conv(self.conv3x3[1], (3, 3))(path2))
        path3 = nn.relu(nn.Conv(self.conv5x5[0], (1, 1))(x))
        path3 = nn.relu(nn.Conv(self.conv5x5[1], (5, 5))(path3))
        path4 = nn.max_pool(x, (3, 3), strides=(1, 1), padding='SAME')
        path4 = nn.relu(nn.Conv(self.pool1x1, kernel_size=(1, 1))(path4))
        return jnp.concatenate([path1, path2, path3, path4], axis=-1)

class AuxilliaryOutput(nn.Module):
    
    @nn.compact
    def __call__(self, x, train):
        x = nn.avg_pool(x, (5, 5), strides=(3, 3))
        x = nn.relu(nn.Conv(128, (1, 1))(x))
        x = x.reshape(x.shape[0], -1)
        x = nn.relu(nn.Dense(1024)(x))
        x = nn.Dropout(0.7, deterministic=not train)(x)
        x = nn.Dense(1000)(x)
        return x

class GoogleNet(nn.Module):
    """Inception-v1"""

    @nn.compact
    def __call__(self, x, train: bool = True):
        x = x / 255.0
        x = nn.relu(nn.Conv(64, (7, 7), strides=2)(x))
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(nn.Conv(64, (1, 1))(x))
        x = nn.relu(nn.Conv(192, (3, 3))(x))
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))
        x = Inception(conv1x1=64, conv3x3=(96, 128), conv5x5=(16, 32), pool1x1=32)(x)
        x = Inception(conv1x1=128, conv3x3=(128, 192), conv5x5=(32, 96), pool1x1=64)(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))
        x = Inception(conv1x1=192, conv3x3=(96, 208), conv5x5=(16, 48), pool1x1=64)(x)

        out1 = AuxilliaryOutput()(x, train)

        x = Inception(conv1x1=160, conv3x3=(112, 224), conv5x5=(24, 64), pool1x1=64)(x)
        x = Inception(conv1x1=128, conv3x3=(128, 256), conv5x5=(24, 64), pool1x1=64)(x)
        x = Inception(conv1x1=112, conv3x3=(144, 288), conv5x5=(32, 64), pool1x1=64)(x)

        out2 = AuxilliaryOutput()(x, train)

        x = Inception(conv1x1=256, conv3x3=(160, 320), conv5x5=(32, 128), pool1x1=128)(x)
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding=((1, 1), (1, 1)))
        x = Inception(conv1x1=256, conv3x3=(160, 320), conv5x5=(32, 128), pool1x1=128)(x)
        x = Inception(conv1x1=384, conv3x3=(192, 384), conv5x5=(48, 128), pool1x1=128)(x)

        x = nn.avg_pool(x, (7, 7), strides=(1, 1))
        x = x.reshape(x.shape[0], -1)
        x = nn.Dropout(0.4, deterministic=not train)(x)
        out3 = nn.Dense(1000)(x)

        return [out1, out2, out3]


if __name__ == '__main__':
    args, writer = get_args_and_writer()
    model = GoogleNet()
    key = jax.random.PRNGKey(args.seed)
    print(model.tabulate(key, jnp.empty(args.input_shape), train=False))

    class TrainState(train_state.TrainState):
        batch_stats: dict
        dropout_key: jax.random.KeyArray

    variables = model.init(key, jnp.empty(args.input_shape), train=False)
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optax.adam(learning_rate=args.learning_rate),
        batch_stats=variables['batch_stats'],
        dropout_key=jax.random.PRNGKey(args.seed),
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
    
    from katacv.utils.imagenet.build_dataset import DatasetBuilder
    ds_builder = DatasetBuilder(args)
    train_ds, train_ds_size = ds_builder.get_dataset(sub_dataset='train')
    val_ds, val_ds_size = ds_builder.get_dataset(sub_dataset='val')

    from functools import partial
    @partial(jax.jit, static_argnames='train')
    def model_step(state: TrainState, x, y, train: bool = True):

        def loss_fn(params):
            logits_list, updates = state.apply_fn(
                {'params': params, 'batch_stats': state.batch_stats},
                x, train=train,
                mutable=['batch_stats'],
                rngs={'dropout': jax.random.fold_in(state.dropout_key, state.step)}
            )
            loss = 0
            for logits, coef in zip(logits_list, (0.3, 0.3, 1.0)):
                loss += -(y * jax.nn.log_softmax(logits)).sum(-1) * coef
            return loss.mean(), (logits, updates)
        
        if train:
            (loss, (logits, updates)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            state = state.apply_gradients(grads=grads)
            state = state.replace(batch_stats=updates['batch_stats'])
        else:
            loss, (logits, _) = loss_fn(state.params, x, y)

        y_cat = jnp.argmax(y, -1)
        top1 = (jnp.argmax(logits, -1) == y_cat).mean()
        top5_pred = jnp.argsort(logits, axis=-1)[:, -5:]
        top5 = jnp.any(top5_pred == y_cat[:, jnp.newaxis], axis=-1).mean()

        return state, loss, top1, top5

    import time
    from tqdm import tqdm

    start_time, global_step = time.time(), 0
    for epoch in range(1,args.total_epochs+1):
        print(f"epoch: {epoch}/{args.total_epochs}")
        logs.reset()
        print("training...")
        for x, y in tqdm(train_ds, total=train_ds_size, desc="Processing"):
        # for x, y in tqdm(train_ds.take(300), total=300, desc="Processing"):
            global_step += 1
            state, *metrics = model_step(state, x.numpy(), y.numpy())
            logs.update(
                ['loss_train', 'accuracy_top1_train', 'accuracy_top5_train', 'SPS', 'SPS_avg'],
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
        for x, y in tqdm(
            val_ds.take(args.val_sample_batch),
            total=args.val_sample_batch,
            desc="Processing"
        ):
            state, *metrics = model_step(state, x.numpy(), y.numpy())
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

    writer.close()