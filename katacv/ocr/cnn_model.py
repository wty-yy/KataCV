from katacv.utils.related_pkgs.jax_flax_optax_orbax import *
from katacv.utils.related_pkgs.utility import *

def mish(x):
    return x * jnp.tanh(jax.nn.softplus(x))

class ConvBlock(nn.Module):
    filters: int
    norm: nn.Module
    act: Callable
    kernel: Sequence[int] = (1, 1)
    strides: Sequence[int] = (1, 1)
    padding: str | Sequence[Sequence[int]] = 'SAME'
    use_norm: bool = True
    use_act: bool = True

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(self.filters, self.kernel, self.strides, self.padding, use_bias=not self.use_norm)(x)
        if self.use_norm: x = self.norm()(x)
        if self.use_act: x = self.act(x)
        return x

class ResBlock(nn.Module):
    conv: nn.Module
    act: Callable

    @nn.compact
    def __call__(self, x):
        residue = x
        n = x.shape[-1] // 2
        x = self.conv(filters=n, kernel=(1,1))(x)
        x = self.conv(filters=2*n, kernel=(3,3), use_act=False)(x)
        return residue + x

class OCR_CNN(nn.Module):
    class_num: int
    stage_size = [1, 1, 2, 2]
    act: Callable = mish

    @nn.compact
    def __call__(self, x, train:bool=True):
        norm = partial(nn.BatchNorm, use_running_average=not train)
        conv = partial(ConvBlock, norm=norm, act=self.act)
        block = partial(ResBlock, conv=conv, act=self.act)
        x = conv(filters=32, kernel=(3,3))(x)
        for i, block_num in enumerate(self.stage_size):
            strides = (2, 2) if i < 2 else (2, 1)
            x = conv(filters=x.shape[-1]*2, kernel=(3,3), strides=strides)(x)
            for _ in range(block_num):
                x = block()(x)
        x = conv(filters=x.shape[-1], kernel=(2,2), padding=((0,0),(0,0)))(x)  # (N,1,15,512)
        x = conv(filters=self.class_num, kernel=(1,1), use_norm=False, use_act=False)(x)
        return x[:,0,...]  # (N,W//4-1,encoder_num)

from katacv.ocr.parser import OCRArgs
def get_learning_rate_fn(args: OCRArgs):
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

class TrainState(train_state.TrainState):
    batch_stats: dict

def get_ocr_cnn_state(args: OCRArgs, verbose=False) -> TrainState:
    args.learning_rate_fn = get_learning_rate_fn(args)
    model = OCR_CNN(class_num=args.class_num)
    key = jax.random.PRNGKey(args.seed)
    if verbose: print(model.tabulate(key, jnp.empty(args.input_shape), train=False))
    variables = model.init(key, jnp.empty(args.input_shape), train=False)
    return TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optax.adam(learning_rate=args.learning_rate_fn),
        batch_stats=variables['batch_stats']
    )

if __name__ == '__main__':
    from katacv.ocr.parser import get_args_and_writer
    args = get_args_and_writer(no_writer=True)
    state = get_ocr_cnn_state(args, verbose=True)
    x = jnp.empty(args.input_shape)
    logits, updates = state.apply_fn(
        {'params': state.params, 'batch_stats': state.batch_stats},
         x, train=False, mutable=['batch_stats']
    )
    print(logits.shape)
