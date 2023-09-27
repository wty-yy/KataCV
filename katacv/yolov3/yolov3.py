# -*- coding: utf-8 -*-
'''
@File    : yolov3.py
@Time    : 2023/09/25 18:37:10
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : 
2023/09/25: Complete yolov3 files:
- `constant.py`: The config constants use in yolov3
- `darknet53.py`: The pre train darknet-53 model.
- `logs.py`: The logs manager.
- `yolov3_model.py`: The `get_yolov3_state()` function build the state with darknet and neck module.

Training on PASCAL VOC:
python katacv/yolov3/yolov3.py --train --model-name YOLOv3-PASCAL --wandb-project-name "PASCAL VOC" \
    --path-dataset-tfrecord "/home/yy/Coding/datasets/PASCAL/tfrecord" \
    --class-num 20

Training on COCO:
python katacv/yolov3/yolov3.py --train --model-name YOLOv3-COCO --wandb-project-name "COCO" \
    --path-dataset-tfrecord "/home/yy/Coding/datasets/COCO/tfrecord" \
    --class-num 80

2023/09/26: complete DEBUG, 
1. Fix gradient bug: forget use `params` in loss_fn.
2. Update mean loss: calcuate mean value based samples.
3. Freeze `darknet` model: stop gradient for the darknet parameters and set `train=False` in darknet.

2023/09/27:
1. update loss func: don't divide by non-zeros number.
2. fix bug: forget load darknet batch stats.
'''
import sys, os
sys.path.append(os.getcwd())
from katacv.utils.related_pkgs.utility import *
from katacv.utils.related_pkgs.jax_flax_optax_orbax import *
from katacv.yolov3.logs import logs

from katacv.yolov3.yolov3_model import TrainState
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
                - y * jax.nn.log_sigmoid(logits)
                - (1-y) * jax.nn.log_sigmoid(-logits)
            )).sum((1,2,3,4)).mean()

        def mse(pred, y, mask):
            return (0.5 * mask * (pred - y) ** 2).sum((1,2,3,4)).mean()

        def ce(logits, y_sparse, mask):
            assert(logits.size//logits.shape[-1] == y_sparse.size)
            C = logits.shape[-1]
            y_onehot = jax.nn.one_hot(y_sparse, num_classes=C)
            pred = -jax.nn.log_softmax(logits)
            return (mask * (pred * y_onehot)).sum((1,2,3,4)).mean()
        
        noobj = target[...,0:1] == 0.0
        obj = target[...,0:1] == 1.0

        ### noobject loss ###
        loss_noobj = bce(logits[...,0:1], 0.0, noobj)
        ### coordinate loss ###
        anchors = anchors.reshape(1, 1, 1, args.B, 2)
        loss_coord = (
            bce(logits[...,1:3], target[...,1:3], obj) +
            # mse(logits[...,3:5], jnp.log(1e-6+target[...,3:5]/anchors), obj)
            mse(jnp.exp(logits[...,3:5])*anchors, target[...,3:5], obj)
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
        ), (loss_noobj, loss_coord, loss_obj, loss_class)

    def loss_fn(params):
        logits, updates = state.apply_fn(
            # Don't use `state.params`!!!
            {'params': {'neck': params['neck'], 'darknet': state.params_darknet}, 'batch_stats': state.batch_stats}
            if args.freeze else
            {'params': params, 'batch_stats': state.batch_stats},
            x, train=train,
            mutable=['batch_stats']
        )
        total_loss = 0
        losses = [0 for _ in range(4)]
        for i in range(len(logits)):
            now_anchors = jnp.array(args.anchors[i*args.B:(i+1)*args.B]) * args.split_sizes[i]
            single_loss, _losses = single_loss_fn(logits[i], y[i], now_anchors)
            total_loss += single_loss
            for i in range(4): losses[i] += _losses[i]
        weight_l2 = 0.5 * sum(
            jnp.sum(x**2) for x in jax.tree_util.tree_leaves(params) if x.ndim > 1
        )
        regular = args.weight_decay * weight_l2
        cost = total_loss + regular
        return cost, (updates, (regular, total_loss, *losses))
    
    if train:
        (cost, (updates, metrics)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        state = state.replace(batch_stats=updates['batch_stats'])
    else:
        cost, (_, metrics) = loss_fn(state.params)
    return state, (cost, *metrics)

if __name__ == '__main__':
    ### Initialize arguments and tensorboard writer ###
    from katacv.yolov3.parser import get_args_and_writer
    args, writer = get_args_and_writer()

    ### Initialize state ###
    from katacv.yolov3.yolov3_model import get_yolov3_state
    state = get_yolov3_state(args, verbose=True)

    ### Load weights ###
    if args.load_id > 0:
        path_load = args.path_cp.joinpath(f"{args.model_name}-{args.load_id:04}")
        assert(path_load.exists())
        with open(path_load, 'rb') as file:
            state = flax.serialization.from_bytes(state, file.read())
        print(f"Successfully load weights from '{str(path_load)}'")
    else:
        weights = ocp.PyTreeCheckpointer().restore(str(args.path_darknet))
        if args.freeze:
            state = state.replace(params_darknet=weights['params']['darknet'])
        else:
            state.params['darknet'] = weights['params']['darknet']
        state.batch_stats['darknet'] = weights['batch_stats']['darknet']
        print(f"Successfully load DarkNet from '{str(args.path_darknet)}'")
    
    ### Save config ###
    save_id = args.load_id + 1
    path_save = args.path_cp.joinpath(f"{args.model_name}-{save_id:04}")
    if path_save.exists():
        print(f"The weights file '{str(path_save)}' already exists, still want to continue? [enter]", end=""); input()

    ### Initialize dataset builder ###
    from katacv.utils.VOC.build_dataset_yolov3 import DatasetBuilder, split_targets
    ds_builder = DatasetBuilder(args)
    train_ds, train_ds_size = ds_builder.get_dataset('train')
    val_ds, val_ds_size = ds_builder.get_dataset('val')

    ### Train and evaluate ###
    start_time, global_step = time.time(), 0
    if args.train:
        for epoch in range(state.step//train_ds_size+1, args.total_epochs + 1):
            print(f"epoch: {epoch}/{args.total_epochs}")
            print("training...")
            logs.reset()
            for x, y in tqdm(train_ds, total=train_ds_size):
                x = x.numpy(); y = split_targets(y, args)
                global_step += 1
                state, metrics = model_step(state, x, y, train=True)
                logs.update(
                    [
                        'cost_train', 'regular_train',
                        'loss_train', 
                        'loss_noobj_train',
                        'loss_coord_train',
                        'loss_obj_train',
                        'loss_class_train',
                    ]
                    , metrics
                )
                if global_step % args.write_tensorboard_freq == 0:
                    logs.update(
                        ['SPS', 'SPS_avg', 'epoch', 'learning_rate'],
                        [
                            args.write_tensorboard_freq/logs.get_time_length(),
                            global_step/(time.time()-start_time),
                            epoch,
                            args.learning_rate_fn(state.step)
                        ]
                    )
                    logs.start_time = time.time()
                    logs.writer_tensorboard(writer, global_step)
            
            logs.reset()
            print("validating...")
            for x, y in tqdm(val_ds, total=val_ds_size):
                x = x.numpy(); y = split_targets(y, args)
                _, metrics = model_step(state, x, y, train=False)
                logs.update(
                    [
                        'epoch',  'learning_rate',
                        'loss_val',
                        'loss_noobj_val',
                        'loss_coord_val',
                        'loss_obj_val',
                        'loss_class_val',
                    ],
                    [
                        epoch, args.learning_rate_fn(state.step),
                        *metrics[-5:]
                    ]
                )
            logs.writer_tensorboard(writer, global_step)

            ### Save weights ###
            if epoch % args.save_weights_freq == 0:
                path_save = args.path_cp.joinpath(f"{args.model_name}-{save_id:04}")
                with open(path_save, 'wb') as file:
                    file.write(flax.serialization.to_bytes(state))
                print(f"Save weights at '{str(path_save)}'")
                save_id += 1
    
    writer.close()
