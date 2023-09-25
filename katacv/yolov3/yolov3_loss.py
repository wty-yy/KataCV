from katacv.utils.related_pkgs.utility import *
from katacv.utils.related_pkgs.jax_flax_optax_orbax import *

from katacv.yolov3.yolov3_model import TrainState
from katacv.yolov3.yolov3 import get_args_and_writer
from katacv.utils.detection import iou

@partial(jax.jit, static_argnames=['train'])
def model_step(
    state: TrainState,
    x: jax.Array,
    y: list,
    train: bool,
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
        def bce(logits, y, mask):
            return (
                mask * (
                - y * jax.nn.log_sigmoid(-logits) -
                (1-y) * jax.nn.log_sigmoid(logits)
            )).mean()

        def mse(pred, y, mask):
            return (0.5 * mask * (pred - y) ** 2).mean()

        def ce(logits, y_sparse, mask):
            assert(logits.size//logits.shape[-1] == y_sparse.size)
            C = logits.shape[-1]
            y_onehot = jax.nn.one_hot(y_sparse, num_classes=args.C)
            pred = -jax.nn.log_softmax(logits)
            return (
                mask * pred * y_onehot
            ).mean()
        
        noobj = target[...,0:1] == 0.0
        obj = target[...,0:1] == 1.0

        ### noobject loss ###
        loss_noobj = bce(logits[...,0:1], 0.0, noobj)
        ### coordinate loss ###
        anchors = anchors.reshape(1, 1, 1, args.B, 2)
        loss_coord = (
            bce(logits[...,1:3], target[...,1:3], obj) +
            mse(logits[...,3:5], jnp.log(1e-6+target[...,3:5]/anchors), obj)
        )
        ### object loss ###
        pred_boxes = jnp.concatenate([
            jax.nn.sigmoid(logits[...,1:3]),
            jnp.exp(logits[...,3:5]) * anchors
        ], axis=-1)
        ious = jax.lax.stop_gradient(iou(pred_boxes, target[...,1:5], keepdim=True))
        loss_obj = bce(logits[...,0:1], ious, obj)
        ### class loss ###
        loss_class = ce(logits[...,5:], target[...,5], obj)

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

if __name__ == '__main__':
    ### Initialize arguments and tensorboard writer ###
    args, writer = get_args_and_writer()

    ### Initialize state ###
    from katacv.yolov3.yolov3_model import get_yolov3_state
    state = get_yolov3_state(args, verbose=False)

    ### Load weights ###
    weights = ocp.PyTreeCheckpointer().restore(str(args.path_darknet))
    state.params['DarkNet_0'] = weights['params']['darknet']
    print(f"Successfully load DarkNet from '{str(args.path_darknet)}'")

    ### Initialize dataset builder ###
    from katacv.utils.VOC.build_dataset_yolov3 import DatasetBuilder, split_targets
    ds_builder = DatasetBuilder(args)
    train_ds, train_ds_size = ds_builder.get_dataset('val')

    ### Train and evaluate ###
    from katacv.yolov3.yolov3_loss import model_step
    start_time, global_step = time.time(), 0
    for epoch in range(1, args.total_epochs + 1):
        print(f"epoch: {epoch}/{args.total_epochs}")
        print("training...")
        bar = tqdm(train_ds, total=train_ds_size)
        loss_mean, count = 0, 0
        for x, y in bar:
            x = x.numpy(); y = split_targets(y, args)
            global_step += 1
            state, loss = model_step(state, x, y, train=True)
            loss_mean += (loss - loss_mean) / (count + 1); count += 1
            bar.set_description(f"loss={loss_mean:.2f}")