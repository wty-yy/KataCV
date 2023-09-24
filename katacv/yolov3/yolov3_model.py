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
        return x

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
        scale1, scale2, scale3 = DarkNet(stage_size=[1,2,8,8,4])(x, train)
        y = block(filters=512)(scale3)
        logits3 = predictor()(y)
        y = conv(filters=256, kernel=(1,1))(y)
        y = jax.image.resize(y, (y.shape[0], 2*y.shape[1], 2*y.shape[2], y.shape[3]), "nearest")
        logits2 = predictor()(block(filters=256)(jnp.concatenate([y, scale2], axis=-1)))
        y = conv(filters=128, kernel=(1,1))(y)
        y = jax.image.resize(y, (y.shape[0], 2*y.shape[1], 2*y.shape[2], y.shape[3]), "nearest")
        logits1 = predictor()(block(filters=128)(jnp.concatenate([y, scale1], axis=-1)))
        return [logits1, logits2, logits3]

from katacv.yolov3.yolov3 import YOLOv3Args

class TrainState(train_state.TrainState):
    batch_stats: dict

def get_yolov3_state(args: YOLOv3Args, verbose=False):
    model = YOLOv3Model(B=args.B, C=args.C)
    key = jax.random.PRNGKey(args.seed)
    if verbose: print(model.tabulate(key, jnp.empty(args.input_shape), train=False))
    variables = model.init(key, jnp.empty(args.input_shape), train=False)
    return TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optax.adam(learning_rate=args.learning_rate),
        batch_stats=variables['batch_stats']
    )

if __name__ == '__main__':
    model = YOLOv3Model(B=3, C=80)
    x = jnp.empty((1,416,416,3))
    print(model.tabulate(jax.random.PRNGKey(0), x))
    params = model.init(jax.random.PRNGKey(0), x, train=False)
    # outputs = model.apply(params, x, train=False)
    logits = model.apply(params, x, train=False)
    print(logits[0].shape, logits[1].shape, logits[2].shape)
