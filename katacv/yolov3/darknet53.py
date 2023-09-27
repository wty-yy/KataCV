# -*- coding: utf-8 -*-
'''
@File    : darknet53.py
@Time    : 2023/09/20 22:52:10
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : Darknet53
2023.09.21. 开始训练
2023.09.22. 完成30epochs的训练，但是效果较差：val-top1 74.45%, val-top5 91.81%比resnet50还差
2023.09.22. 重新训练50epochs, remove batch normalize bias and use mish as activate function
'''
import sys, os
sys.path.append(os.getcwd())

from katacv.utils.related_pkgs.utility import *
from katacv.utils.related_pkgs.jax_flax_optax_orbax import *  # jax, jnp, flax, nn, train_state, optax

ModuleDef = Any

class ConvBlock(nn.Module):
    filters: int
    norm: ModuleDef
    act: Callable
    kernel: Tuple[int, int] = (1, 1)
    strides: Tuple[int, int]  = (1, 1)
    use_norm: bool = True
    use_act: bool = True

    @nn.compact
    def __call__(self, x):
        # do not use bias when norm is activate: https://arxiv.org/pdf/1502.03167.pdf
        x = nn.Conv(self.filters, self.kernel, self.strides, use_bias=not self.use_norm)(x)
        if self.use_norm: x = self.norm()(x)
        if self.use_act: x = self.act(x)
        return x

class ResBlock(nn.Module):
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    
    @nn.compact
    def __call__(self, x):
        n = x.shape[-1] // 2
        residue = x
        x = self.conv(filters=n, kernel=(1,1))(x)
        x = self.conv(filters=2*n, kernel=(3,3), use_act=False)(x)
        return self.act(x + residue)

def mish(x):  # too slow, one epoch 52mins, leaky_relu epoch 46mins
    return x * jnp.tanh(jnp.log(1+jnp.exp(x)))

class DarkNet(nn.Module):
    stage_size: Sequence[int]
    block_cls: ModuleDef = ResBlock
    # act: Callable = mish
    act: Callable = lambda x: nn.leaky_relu(x, 0.1)

    @nn.compact
    def __call__(self, x, train: bool = True):
        norm = partial(nn.BatchNorm, use_running_average=not train)
        conv = partial(ConvBlock, norm=norm, act=self.act)
        block = partial(self.block_cls, conv=conv, norm=norm, act=self.act)
        x = conv(filters=32, kernel=(3,3))(x)
        outputs = []
        for i, block_size in enumerate(self.stage_size):
            x = conv(filters=x.shape[-1]*2, kernel=(3,3), strides=(2,2))(x)
            for _ in range(block_size):
                x = block()(x)
            if i > 1: outputs.append(x)
        return outputs

class PreTrain(nn.Module):
    darknet: ModuleDef
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        x = self.darknet(x, train)[-1]
        x = jnp.mean(x, (1, 2))  # same as nn.avg_pool(x, (7, 7))
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
        'SPS_avg': MeanMetric(),
        'learning_rate': 0,
    },
    folder2name={
        'train/metrics': ['loss_train', 'accuracy_top1_train', 'accuracy_top5_train'],
        'val/metrics': ['loss_val', 'accuracy_top1_val', 'accuracy_top5_val'],
        'charts': ['SPS', 'SPS_avg', 'epoch', 'learning_rate']
    }
)

def get_args_and_writer(no_writer=False):
    from katacv.utils.parser import Parser, Path, cvt2Path, datetime
    parser = Parser(model_name="DarkNet53", wandb_project_name="Imagenet2012")
    # Imagenet dataset
    parser.add_argument("--path-dataset-tfrecord", type=cvt2Path, default=Path("/media/yy/Data/dataset/imagenet/tfrecord"),
        help="the path of the tfrecord dataset directory")
    parser.add_argument("--image-size", type=int, default=224,
        help="the input image size of the model")
    parser.add_argument("--image-center-crop-padding-size", type=int, default=32,
        help="the padding size of the center crop of the origin image")
    parser.add_argument("--batch-size", type=int, default=128,
        help="the size of each batch")
    parser.add_argument("--shuffle-size", type=int, default=128*16,
        help="the shuffle size of the dataset")
    # Hyper-parameters
    parser.add_argument("--weight-decay", type=float, default=1e-4,
        help="the coef of the weight penalty")
    parser.add_argument("--momentum", type=float, default=0.9,
        help="the momentum of the SGD optimizer")
    parser.add_argument("--total-epochs", type=int, default=50,
        help="the total epochs of the training")
    parser.add_argument("--learning-rate", type=float, default=0.05,
        help="the learning rate of the optimizer")
    parser.add_argument("--warmup-epochs", type=int, default=3,
        help="the number of warming up epochs")
    
    args = parser.get_args()
    assert(args.total_epochs > args.warmup_epochs)
    args.run_name = f"{args.model_name}__load_{args.load_id}__warmup_lr_{args.learning_rate}__batch_{args.batch_size}__{datetime.datetime.now().strftime(r'%Y%m%d_%H%M%S')}".replace("/", "-")
    args.input_shape = (args.batch_size, args.image_size, args.image_size, 3)
    if no_writer: return args
    writer = parser.get_writer(args)
    return args, writer

class TrainState(train_state.TrainState):
    batch_stats: dict

@partial(jax.jit, static_argnames='train')
def model_step(state: TrainState, x, y, train: bool = True):

    def loss_fn(params):
        logits, updates = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            x, train=train,
            mutable=['batch_stats'],
        )
        loss = -(y * jax.nn.log_softmax(logits)).sum(-1).mean()
        weight_l2 = 0.5 * sum(
            jnp.sum(x**2) for x in jax.tree_util.tree_leaves(params) if x.ndim > 1  # no bias
        )
        loss += args.weight_decay * weight_l2
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

def get_learning_rate_fn(args):
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=args.learning_rate,
        transition_steps=args.warmup_epochs * args.steps_pre_epoch
    )
    cosine_epoch = args.total_epochs - args.warmup_epochs
    cosine_fn = optax.cosine_decay_schedule(
        init_value=args.learning_rate,
        decay_steps=cosine_epoch * args.steps_pre_epoch
    )
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[args.warmup_epochs * args.steps_pre_epoch]
    )
    return schedule_fn

def get_pretrain_state(args=None, verbose=False):
    from katacv.utils.imagenet.build_dataset import DATASET_SIZE
    train_ds_size = DATASET_SIZE['train']
    args.steps_pre_epoch = train_ds_size // args.batch_size
    args.learning_rate_fn = get_learning_rate_fn(args)
    model_name, seed, input_shape, tx = (
        args.model_name,
        args.seed,
        args.input_shape,
        optax.sgd(learning_rate=args.learning_rate_fn, momentum=args.momentum, nesterov=True)
    )
    model = PreTrain(darknet=DarkNet(stage_size=[1,2,8,8,4], name='darknet'))
    key = jax.random.PRNGKey(seed)
    if verbose:
        print(model.tabulate(key, jnp.empty(input_shape)))
    variables = model.init(key, jnp.empty(input_shape))
    return TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=tx,
        batch_stats=variables['batch_stats'],
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
        for epoch in range(state.step//args.steps_pre_epoch+1, args.total_epochs+1):
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
                        ['SPS', 'SPS_avg', 'epoch', 'learning_rate'],
                        [
                            args.write_tensorboard_freq/logs.get_time_length(),
                            global_step/(time.time()-start_time),
                            epoch, args.learning_rate_fn(state.step)
                        ]
                    )
                    logs.start_time = time.time()
                    logs.writer_tensorboard(writer, global_step)

            logs.reset()
            print("validating...")
            for x, y in tqdm(val_ds, total=val_ds_size, desc="Processing"):
                state, *metrics = model_step(state, x.numpy(), y.numpy(), train=False)
                logs.update(
                    ['loss_val', 'accuracy_top1_val', 'accuracy_top5_val', 'epoch', 'learning_rate'],
                    metrics + [epoch, args.learning_rate_fn(state.step)]
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