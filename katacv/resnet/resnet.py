# -*- coding: utf-8 -*-
'''
@File    : resnet.py
@Time    : 2023/09/17 08:52:34
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : 
refer: https://github.com/google/flax/blob/main/examples/imagenet/models.py
2023.9.18. chagne to use sgd and warming up learning rate
'''
from katacv.utils.related_pkgs.utility import *
from katacv.utils.related_pkgs.jax_flax_optax_orbax import *  # jax, jnp, flax, nn, train_state, optax

ModuleDef = Any

class BottleneckResNetBlock(nn.Module):
    filters: int
    conv: ModuleDef
    norm: ModuleDef
    act: Callable
    strides: Tuple[int, int]  = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.conv(self.filters, (1, 1))(x)  # change a variable name, such as y !!
        y = self.act(self.norm()(y))
        y = self.conv(self.filters, (3, 3), self.strides)(y)
        y = self.act(self.norm()(y))
        y = self.conv(self.filters * 4, (1, 1))(y)
        y = self.norm(scale_init=nn.initializers.zeros_init())(y)
        if residual.shape != y.shape:
            residual = self.conv(
                self.filters * 4, (1, 1), self.strides, name='conv_proj'
            )(residual)
            residual = self.norm(name='norm_proj')(residual)
        return self.act(y + residual)

class ResNet(nn.Module):
    stage_size: Sequence[int]
    block_cls: ModuleDef = BottleneckResNetBlock
    filters: int = 64
    conv: ModuleDef = nn.Conv
    act: Callable = nn.relu

    @nn.compact
    def __call__(self, x, train: bool = True):
        conv = partial(self.conv, use_bias=False)
        norm = partial(
            nn.BatchNorm,
            use_running_average=not train,
            momentum=0.9,
            epsilon=1e-5
        )
        x = conv(self.filters, (7, 7), strides=(2, 2), padding=((3, 3), (3, 3)), name='conv_init')(x)
        x = self.act(norm(name='bn_init')(x))
        x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')
        for i, block_size in enumerate(self.stage_size):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = self.block_cls(
                    self.filters * 2 ** i,
                    strides=strides,
                    conv=conv,
                    norm=norm,
                    act=self.act
                )(x)
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
        'learning_rate': MeanMetric()
    },
    folder2name={
        'train/metrics': ['loss_train', 'accuracy_top1_train', 'accuracy_top5_train'],
        'val/metrics': ['loss_val', 'accuracy_top1_val', 'accuracy_top5_val'],
        'charts': ['SPS', 'SPS_avg', 'epoch', 'learning_rate']
    }
)

def get_args_and_writer():
    from katacv.utils.parser import Parser, Path, cvt2Path
    parser = Parser(model_name="ResNet50", wandb_project_name="Imagenet2012")
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
    
    args, writer = parser.get_args_and_writer()
    assert(args.total_epochs > args.warmup_epochs)
    args.input_shape = (args.batch_size, args.image_size, args.image_size, 3)
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

model_cls = {
    'ResNet50': partial(ResNet, stage_size=(3,4,6,3)),
    'ResNet101': partial(ResNet, stage_size=(3,4,23,3)),
    'ResNet152': partial(ResNet, stage_size=(3,8,36,3)),
    'ResNet200': partial(ResNet, stage_size=(3,24,36,3)),
}

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
    model_name, seed, input_shape, lr_fn = (
        ('ResNet50', 0, (1,224,224,3), 0.1) if args is None else
        (args.model_name, args.seed, args.input_shape, args.learning_rate_fn)
    )
    model = model_cls[model_name]()
    key = jax.random.PRNGKey(seed)
    if verbose:
        print(model.tabulate(key, jnp.empty(input_shape)))
    variables = model.init(key, jnp.empty(input_shape))
    return TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optax.sgd(learning_rate=lr_fn, momentum=args.momentum, nesterov=True),
        batch_stats=variables['batch_stats'],
    )

if __name__ == '__main__':
    args, writer = get_args_and_writer()
    
    from katacv.utils.imagenet.build_dataset import ImagenetBuilder
    ds_builder = ImagenetBuilder(args)
    train_ds, train_ds_size = ds_builder.get_dataset(sub_dataset='train')
    val_ds, val_ds_size = ds_builder.get_dataset(sub_dataset='val')
    args.steps_pre_epoch = train_ds_size // args.batch_size

    args.learning_rate_fn = get_learning_rate_fn(args)
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

    import time
    from tqdm import tqdm

    start_time, global_step = time.time(), 0
    if args.train:
        for epoch in range(state.step//args.steps_pre_epoch, args.total_epochs+1):
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