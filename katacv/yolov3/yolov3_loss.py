from katacv.utils.related_pkgs.utility import *
from katacv.utils.related_pkgs.jax_flax_optax_orbax import *

from katacv.yolov3.yolov3_model import TrainState
from katacv.yolov3.yolov3 import YOLOv3Args
from katacv.utils.detection import iou
@partial(jax.jit, static_argnames=['train', 'args'])
def model_step(
    state: TrainState,
    x: jax.Array,
    y: list,
    train: bool,
    args: YOLOv3Args,
):
    def single_loss_fn(logits, target, anchors):
        """
        ### Shape
        logits.shape=(N,S,S,B,5+C)  last=(c,x,y,w,h,{pr})
            {pr} is a probability distribution
        target.shape=(N,S,S,B,6)    last=(c,x,y,w,h,cls)
        ### Loss
        - (c,x,y):  binary cross-entropy (BCE)
        - (w,h):    mean square error (MSE)
        - {pr}:     corss entropy (CE)
        """
        def bce(logits, y):
            return -(
                y * jax.nn.softplus(-logits) +
                (1-y) * jax.nn.softplus(logits)
            ).mean()

        def mse(pred, y):
            return 0.5 * ((pred - y) ** 2).mean()

        def ce(logits, y_sparse):
            assert(logits.size//logits.shape[-1] == y_sparse.size)
            C = logits.shape[-1]
            log = -jax.nn.log_softmax(logits).reshape(-1, C)
            return log[jnp.arange(log.shape[0]), y_sparse.reshape(-1)].mean()
        
        noobj = target[...,0] == 0.0
        obj = target[...,0] == 1.0

        ### noobject loss ###
        loss_noobj = bce(logits[...,0:1][noobj], 0.0)
        ### coordinate loss ###
        anchors = anchors.reshape(1, 1, 1, args.B, 2)
        loss_coord = (
            bce(logits[...,1:3][noobj], target[...,1:3][noobj]) +
            mse(logits[...,3:5], jnp.log(1e-6+target[...,3:5]/anchors))
        )
        ### object loss ###
        pred_boxes = jnp.concatenate([
            jax.nn.sigmoid(logits[...,1:3]),
            jax.nn.exp(logits[...,3:5]) * anchors
        ], axis=-1)
        ious = jax.lax.stop_gradient(iou(pred_boxes[obj], target[...,1:5][obj], keepdim=True))
        loss_obj = bce(logits[...,0:1][obj], ious[obj])
        ### class loss ###
        loss_class = ce(logits[...,5:][obj], target[...,5][obj])

        return (
            args.coef_noobj * loss_noobj + 
            args.coef_coord * loss_coord + 
            args.coef_obj   * loss_obj +
            args.coef_class * loss_class
        )

    def loss_fn(params):
        logits, updates = state.apply_fn(
            {'params': state.params, 'batch_stats': state.batch_stats},
            x, train=train,
            mutable=['batch_stats']
        )
        loss = 0
        for i in range(len(logits)):
            anchors = jnp.array(args.anchors[i*args.B:(i+1)*args.B]) * args.split_sizes[i]
            loss += single_loss_fn(logits[i], y[i], anchors)
        weight_l2 = 0.5 * sum(
            jnp.sum(x**2) for x in jax.tree_util.tree_leaves(params) if x.ndim > 1
        )
        loss += args.weight_decay * weight_l2
        return loss, updates
    
    if train:
        (loss, updates), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        state = state.replace(batch_stats=updates['batch_stats'])
    else:
        loss, _ = loss_fn(state.params)
    return state, loss
