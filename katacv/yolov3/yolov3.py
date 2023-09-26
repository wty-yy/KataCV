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
'''
import sys, os
sys.path.append(os.getcwd())
from katacv.utils.related_pkgs.utility import *
from katacv.utils.related_pkgs.jax_flax_optax_orbax import *
from katacv.yolov3.logs import logs
import katacv.yolov3.constant as const

from katacv.utils.parser import Parser, CVArgs, cvt2Path, SummaryWriter

class YOLOv3Args(CVArgs):
    split_sizes: int
    bounding_box: int;  B: int
    class_num: int;     C: int
    anchors: List[Tuple[int, int]]
    iou_ignore_threshold: float
    path_darknet: Path
    coef_noobj: float
    coef_coord: float
    coef_obj:   float
    coef_class: float

def get_args_and_writer(no_writer=False) -> tuple[YOLOv3Args, SummaryWriter]:
    # parser = Parser(model_name="YOLOv3-COCO", wandb_project_name="COCO")
    parser = Parser(model_name="YOLOv3-PASCAL", wandb_project_name="PASCAL VOC")
    ### Dataset config ###
    parser.add_argument("--path-dataset-tfrecord", type=cvt2Path, default=const.path_dataset_tfrecord,
        help="the tfrecords folder of the VOC dataset (COCO or PASCAL VOC)")
    parser.add_argument("--batch-size", type=int, default=const.batch_size,
        help="the batch size of the model")
    parser.add_argument("--shuffle-size", type=int, default=const.shuffle_size,
        help="the shuffle size of the dataset")
    parser.add_argument("--image-size", type=int, default=const.image_size,
        help="the image size of the model input")
    parser.add_argument("--split-sizes", type=int, default=const.split_sizes,
        help="the split size of the cells")
    parser.add_argument("--bounding-box", type=int, default=const.bounding_box,
        help="the number of bounding box in each cell (relative to the anchor boxes)")
    parser.add_argument("--class-num", type=int, default=const.class_num,
        help="the number of the classes (labels)")
    parser.add_argument("--anchors", default=const.anchors,
        help="the anchors for different split sizes")
    parser.add_argument("--iou-ignore-threshold", type=float, default=const.iou_ignore_threshold,
        help="the threshold of iou, \
        if the iou between anchor box and \
        target box (not the biggest iou one) is bigger than `iou_ignore_threshold`, \
        then it will be ignore (this example will not add in loss).")
    ### Loss config ###
    parser.add_argument("--coef-noobj", type=float, default=const.coef_noobj,
        help="the coef of the no object loss")
    parser.add_argument("--coef-coord", type=float, default=const.coef_coord,
        help="the coef of the coordinate loss")
    parser.add_argument("--coef-obj", type=float, default=const.coef_obj,
        help="the coef of the object loss")
    parser.add_argument("--coef-class", type=float, default=const.coef_class,
        help="the coef of the class loss")
    ### Training config ###
    parser.add_argument("--total-epochs", type=int, default=const.total_epochs,
        help="the total epochs of the training")
    parser.add_argument("--learning-rate", type=float, default=const.learning_rate,
        help="the learning rate of the optimizer")
    parser.add_argument("--weight-decay", type=float, default=const.weight_decay,
        help="the coef of the weight penalty")
    ### Model config ###
    parser.add_argument("--path-darknet", type=cvt2Path, default=const.path_darknet)
    parser.add_argument("--darknet-id", type=int, default=const.darknet_id,
        help="the weights id of the darknet model")

    args = parser.get_args()
    args.write_tensorboard_freq = 10
    args.B, args.C = args.bounding_box, args.class_num
    args.path_darknet = args.path_darknet.joinpath(f"DarkNet53-{args.darknet_id:04}-lite")
    args.input_shape = (args.batch_size, args.image_size, args.image_size, 3)
    if no_writer: return args

    writer = parser.get_writer(args)
    return args, writer

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
            return ((
                mask * (
                - y * jax.nn.log_sigmoid(logits)
                - (1-y) * jax.nn.log_sigmoid(-logits)
            )).sum((1,2,3,4)) / mask.sum((1,2,3,4))).mean()

        def mse(pred, y, mask):
            return ((0.5 * mask * (pred - y) ** 2).sum((1,2,3,4)) / mask.sum((1,2,3,4))).mean()

        def ce(logits, y_sparse, mask):
            assert(logits.size//logits.shape[-1] == y_sparse.size)
            C = logits.shape[-1]
            y_onehot = jax.nn.one_hot(y_sparse, num_classes=C)
            pred = -jax.nn.log_softmax(logits)
            return ((mask * (pred * y_onehot)).sum((1,2,3,4)) / mask.sum((1,2,3,4))).mean()
        
        noobj = target[...,0:1] == 0.0
        obj = target[...,0:1] == 1.0

        ### noobject loss ###
        loss_noobj = bce(logits[...,0:1], 0.0, noobj)
        ### coordinate loss ###
        anchors = anchors.reshape(1, 1, 1, args.B, 2)
        loss_coord = (
            bce(logits[...,1:3], target[...,1:3], obj) +
            mse(logits[...,3:5], jnp.log(1e-6+target[...,3:5]/anchors), obj)
            # mse(jnp.exp(logits[...,3:5])*anchors, target[...,3:5], obj)
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
            # Don't use `state.params`!!!
            {'params': {'neck': params, 'darknet': state.params_darknet}, 'batch_stats': state.batch_stats},
            x, train=train,
            mutable=['batch_stats']
        )
        loss = 0
        for i in range(len(logits)):
            now_anchors = jnp.array(args.anchors[i*args.B:(i+1)*args.B]) * args.split_sizes[i]
            loss += single_loss_fn(logits[i], y[i], now_anchors)
        weight_l2 = 0.5 * sum(
            jnp.sum(x**2) for x in jax.tree_util.tree_leaves(params) if x.ndim > 1
        )
        regular = args.weight_decay * weight_l2
        cost = loss + regular
        return cost, (updates, (loss, regular))
    
    if train:
        (cost, (updates, metrics)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        state = state.replace(batch_stats=updates['batch_stats'])
    else:
        cost, (_, metrics) = loss_fn(state.params)
    return state, (cost, *metrics)

if __name__ == '__main__':
    ### Initialize arguments and tensorboard writer ###
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
        state.params['darknet'] = weights['params']['darknet']
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
                logs.update(['cost_train', 'loss_train', 'regular_train'], metrics)
                if global_step % args.write_tensorboard_freq == 0:
                    logs.update(
                        ['SPS', 'SPS_avg', 'epoch'],
                        [args.write_tensorboard_freq/logs.get_time_length(), global_step/(time.time()-start_time), epoch]
                    )
                    logs.start_time = time.time()
                    logs.writer_tensorboard(writer, global_step)
            
            logs.reset()
            print("validating...")
            for x, y in tqdm(val_ds, total=val_ds_size):
                x = x.numpy(); y = split_targets(y, args)
                _, metrics = model_step(state, x, y, train=False)
                logs.update(['loss_val', 'epoch'], [metrics[1], epoch])
            logs.writer_tensorboard(writer, global_step)

            ### Save weights ###
            if epoch % args.save_weights_freq == 0:
                path_save = args.path_cp.joinpath(f"{args.model_name}-{save_id:04}")
                with open(path_save, 'wb') as file:
                    file.write(flax.serialization.to_bytes(state))
                print(f"Save weights at '{str(path_save)}'")
                save_id += 1
    
    writer.close()
