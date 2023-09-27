# -*- coding: utf-8 -*-
'''
@File    : yolov3_model.py
@Time    : 2023/09/26 20:38:44
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    :
2023/09/24: complete `neck` module.
2023/09/26: freeze `darknet` module.
'''

if __name__ == '__main__':
    pass

from katacv.utils.related_pkgs.utility import *
from katacv.utils.related_pkgs.jax_flax_optax_orbax import *
from katacv.yolov3.darknet53 import DarkNet, ConvBlock

class YOLOBlock(nn.Module):
    filters: int
    conv: nn.Module

    @nn.compact
    def __call__(self, x):
        for _ in range(2):
            x = self.conv(filters=self.filters, kernel=(1,1))(x)
            x = self.conv(filters=self.filters*2, kernel=(3,3))(x)
        x = self.conv(filters=self.filters, kernel=(1,1))(x)
        return x

class ScalePredict(nn.Module):
    conv: nn.Module
    B: int
    C: int

    @nn.compact
    def __call__(self, x):
        n = x.shape[-1]
        x = self.conv(filters=2*n, kernel=(3,3))(x)
        x = self.conv(filters=self.B*(self.C+5), kernel=(1,1), use_norm=False, use_act=False)(x)
        return x.reshape(x.shape[0], x.shape[1], x.shape[2], self.B, 5 + self.C)
        # (N, S, S, B, 5 + C)


class Neck(nn.Module):
    norm: nn.Module
    conv: nn.Module
    block: nn.Module
    predictor: nn.Module

    @nn.compact
    def __call__(self, scales):
        scale1, scale2, scale3 = scales
        y = self.block(filters=512)(scale3)
        output3 = self.predictor()(y)
        y = self.conv(filters=256, kernel=(1,1))(y)
        y = jax.image.resize(y, (y.shape[0], 2*y.shape[1], 2*y.shape[2], y.shape[3]), "nearest")
        output2 = self.predictor()(self.block(filters=256)(jnp.concatenate([y, scale2], axis=-1)))
        y = self.conv(filters=128, kernel=(1,1))(y)
        y = jax.image.resize(y, (y.shape[0], 2*y.shape[1], 2*y.shape[2], y.shape[3]), "nearest")
        output1 = self.predictor()(self.block(filters=128)(jnp.concatenate([y, scale1], axis=-1)))
        return [output1, output2, output3]

class YOLOv3Model(nn.Module):
    B: int  # number of bouding boxes
    C: int  # number of target classes
    act: Callable = lambda x: nn.leaky_relu(x, 0.1)

    @nn.compact
    def __call__(self, x, train: bool = True):
        norm = partial(nn.BatchNorm, use_running_average=not train)
        conv = partial(ConvBlock, norm=norm, act=self.act)
        block = partial(YOLOBlock, conv=conv)
        predictor = partial(ScalePredict, conv=conv, B=self.B, C=self.C)
        scales = DarkNet(stage_size=[1,2,8,8,4], name='darknet')(x, train)
        outputs = Neck(norm=norm, conv=conv, block=block, predictor=predictor, name='neck')(scales)
        return outputs

class TrainState(train_state.TrainState):
    params_darknet: dict
    batch_stats: dict

def get_learning_rate_fn(args):
    """
    - `args.learning_rate`: the target warming up learning rate.
    - `args.warmup_epochs`: the epochs get to the target learning rate.
    - `args.steps_per_epoch`: number of the steps to each per epoch.
    """
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=args.learning_rate,
        transition_steps=args.warmup_epochs * args.steps_per_epoch
    )
    cosine_epoch = args.total_epochs - args.warmup_epochs
    cosine_fn = optax.cosine_decay_schedule(
        init_value=args.learning_rate,
        decay_steps=cosine_epoch * args.steps_per_epoch
    )
    schedule_fn = optax.join_schedules(
        schedules=[warmup_fn, cosine_fn],
        boundaries=[args.warmup_epochs * args.steps_per_epoch]
    )
    return schedule_fn

from katacv.yolov3.parser import YOLOv3Args
def get_yolov3_state(args: YOLOv3Args, verbose=False) -> TrainState:
    from katacv.yolov3.constant import train_ds_size
    args.steps_per_epoch = train_ds_size // args.batch_size
    args.learning_rate_fn = get_learning_rate_fn(args)
    model = YOLOv3Model(B=args.B, C=args.C)
    key = jax.random.PRNGKey(args.seed)
    if verbose: print(model.tabulate(key, jnp.empty(args.input_shape), train=False))
    variables = model.init(key, jnp.empty(args.input_shape), train=False)
    params = {'neck': variables['params']['neck']}
    if not args.freeze:
        params['darknet'] = variables['params']['darknet']
    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        params_darknet=variables['params']['darknet'] if args.freeze else None,
        tx=optax.adam(learning_rate=args.learning_rate_fn),
        batch_stats=variables['batch_stats']
    )

if __name__ == '__main__':
    from katacv.yolov3.yolov3 import get_args_and_writer
    args = get_args_and_writer(no_writer=True)
    state = get_yolov3_state(args, verbose=True)
    x = jnp.empty((1,416,416,3))
    logits, updates = state.apply_fn(
        {'params': {'neck': state.params, 'darknet': state.params_darknet}, 'batch_stats': state.batch_stats},
        x, train=False,
        mutable=['batch_stats']
    )
    print(state.batch_stats.keys())
    print(logits[0].shape, logits[1].shape, logits[2].shape)
