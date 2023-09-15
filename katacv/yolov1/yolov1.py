import jax, jax.numpy as jnp
import flax, flax.linen as nn
from flax.training import train_state
import optax
from katacv.utils.imagenet.build_dataset import ImagenetBuilder
from yolov1_pretrain import Darknet, ConvBlock, partial
from katacv.utils.logs import Logs, MeanMetric
from pathlib import Path
from typing import Callable

logs = Logs(
    init_logs={
        'loss_train': MeanMetric(),
        'loss_coord_train': MeanMetric(),
        'loss_conf_train': MeanMetric(),
        'loss_noobj_train': MeanMetric(),
        'loss_class_train': MeanMetric(),
        'mAP_train': MeanMetric(),
        'coco_mAP_train': MeanMetric(),

        'loss_val': MeanMetric(),
        'loss_coord_val': MeanMetric(),
        'loss_conf_val': MeanMetric(),
        'loss_noobj_val': MeanMetric(),
        'loss_class_val': MeanMetric(),
        'mAP_val': MeanMetric(),
        'coco_mAP_val': MeanMetric(),

        'epoch': 0,
        'SPS': MeanMetric(),
        'SPS_avg': MeanMetric()
    },
    folder2name={
        'train/metrics': ['loss_train', 'loss_coord_train', 'loss_conf_train', 'loss_noobj_train', 'loss_class_train', 'mAP_train', 'coco_mAP_train'],
        'val/metrics': ['loss_val', 'loss_coord_val', 'loss_conf_val', 'loss_noobj_val', 'loss_class_val', 'mAP_val', 'coco_mAP_val'],
        'charts': ['SPS', 'SPS_avg', 'epoch']
    }
)

from katacv.utils.parser import Parser, cvt2Path, SummaryWriter, CVArgs
class YoloV1Args(CVArgs):
    split_size: int
    S: int
    bound_box: int
    B: int
    class_num: int
    C: int
    path_pretrain: Path
    coef_coord: int
    coef_noobj: int

def get_args_and_writer() -> tuple[YoloV1Args, SummaryWriter]:
    parser = Parser()
    # VOC Dataset config
    parser.add_argument("--path-dataset-tfrecord", type=cvt2Path, default=Path("/home/wty/Coding/datasets/VOC/tfrecord"),
        help="the tfrecord of the PASCAL VOC dataset")
    parser.add_argument("--batch-size", type=int, default=64,
        help="the batch size of the model")
    parser.add_argument("--shuffle-size", type=int, default=64*16,
        help="the shuffle size of the dataset")
    parser.add_argument("--image-size", type=int, default=448,
        help="the image size of the model input")
    parser.add_argument("--split-size", type=int, default=7,
        help="the split size of the cells")
    parser.add_argument("--class-num", type=int, default=20,
        help="the number of the classes (labels)")
    parser.add_argument("--bounding-box", type=int, default=2,
        help="the number of bounding box in each cell")
    parser.add_argument("--coef-coord", type=float, default=5.0,
        help="the coef of the coordinate loss")
    parser.add_argument("--coef-noobj", type=float, default=0.5,
        help="the coef of the no object loss")
    # VOC pre-train model
    args, writer = parser.get_args_and_writer()
    args.S, args.B, args.C = args.split_size, args.bounding_box, args.class_num
    args.path_pretrain = args.path_logs.joinpath("YoloV1PreTrain-checkpoints")
    args.input_shape = (args.batch_size, args.image_size, args.image_size, 3)
    return args, writer

class YoloV1(nn.Module):
    activation: Callable = lambda x: nn.leaky_relu(x, negative_slope=0.1)
    S: int = 7  # split_size
    B: int = 2  # bounding_box_num
    C: int = 20 # class
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        x = Darknet(activation=self.activation)(x, train)
        norm = partial(nn.BatchNorm, use_running_average=not train)
        conv = partial(ConvBlock, activation=self.activation, norm=norm)
        x = conv(features=1024, kernel_size=(3,3))(x)
        x = conv(features=1024, kernel_size=(3,3), strides=(2,2), padding=((1,1),(1,1)))(x)
        x = conv(features=1024, kernel_size=(3,3))(x)
        x = conv(features=1024, kernel_size=(3,3))(x)
        x = x.reshape(x.shape[0], -1)
        x = self.activation(nn.Dense(2048)(x))  # small ??
        x = nn.Dropout(0.5, deterministic=not train)(x)
        x = nn.Dense(self.S*self.S*(self.C+5*self.B))(x)
        return x

class TrainState(train_state.TrainState):
    batch_stats: dict
    dropout_key: jax.random.KeyArray

from katacv.utils.detection import slice_by_idxs, iou
@jax.jit
def model_step(state: TrainState, x, y, train: bool = True):

    def loss_fn(params):
        cells, updates = state.apply_fn(
            {'params': params, 'batch_stats': state.batch_stats},
            x, train=train,
            mutable=['batch_stats'],
            rngs={'dropout': jax.random.fold_in(state.dropout_key, state.step)}
        )
        cells = cells.reshape(-1, args.S, args.S, args.C+5*args.B)   # NxSxSx(C+5*B)
        target_boxes = y[...,args.C:]    # NxSxSx5
        exist_obj = target_boxes[...,0]  # NxSxS
        # coordinate loss
        ious = jnp.concatenate([
            iou(cells[...,args.C+i*5:args.C+(i+1)*5], target_boxes, scale=[1,1,args.S,args.S], keepdim=True)
            for i in range(args.B)
            ], axis=-1)  # NxSxSxB
        best_idxs = jnp.argmax(ious, axis=-1, keepdims=True)         # NxSxSx1
        best_boxes = slice_by_idxs(cells, best_idxs, 5)              # NxSxSx5, last dim: (c,x,y,w,h)
        loss_coord = 0.5 * (exist_obj * (
            (best_boxes[...,1]-target_boxes[...,1])**2 +  # x
            (best_boxes[...,2]-target_boxes[...,2])**2 +  # y
            (jnp.sqrt(jnp.abs(best_boxes[...,3]))-jnp.sqrt(target_boxes[...,3]))**2 +  # sqrt(w)
            (jnp.sqrt(jnp.abs(best_boxes[...,4]))-jnp.sqrt(target_boxes[...,4]))**2    # sqrt(h)
        )).sum([1,2]).mean()
        # confidence loss
        loss_conf = 0.5 * (exist_obj * (best_boxes[...,0]-target_boxes[...,0])**2).sum([1,2]).mean()
        # noobject loss
        loss_noobj = 0
        for i in range(args.B):  # if no object in cell, decrease all the box confidence.
            tmp = 0.5 * ((1-exist_obj) * (cells[...,args.C+i*5]-y[...,args.C])**2).sum([1,2]).mean()
            loss_noobj += (tmp - loss_noobj) / (i+1)
        # class loss
        loss_class = 0.5 * (exist_obj * ((jax.nn.softmax(cells[...,:args.C])-y[...,:args.C])**2).sum(-1)).sum([1,2]).mean()

        loss = args.coef_coord * loss_coord + \
               loss_conf + \
               args.coef_noobj * loss_noobj + \
               loss_class
        return loss, (updates, (loss_coord, loss_conf, loss_noobj, loss_class, cells))
    
    if train:
        (loss, (updates, metrics)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        state = state.apply_gradients(grads=grads)
        state = state.replace(batch_stats=updates['batch_stats'])
    else:
        loss, (_, metrics) = loss_fn(state.params)
    return state, (loss, *metrics)

from katacv.utils.detection import nms, mAP, coco_mAP
def get_best_boxes_and_classes(cells):
    """
    Get the best confidence boxes and classes in cells.
    params::cells.shape=(S,S,C+5*B)
    return::boxes.shape=(SxS,6), the last dim: (c,x,y,w,h,cls)
    """
    conf_idxs = jnp.argmax(cells[...,args.C+jnp.arange(args.B)*5], axis=-1)
    conf_boxes = slice_by_idxs(cells, conf_idxs, 5)  # SxSx5
    boxes = []
    for i in range(args.S):
        for j in range(args.S):
            pred_class = jnp.argmax(cells[i,j,:args.C]); pred_prob = cells[i,j,pred_class]
            conf = conf_boxes[i,j,0] * pred_prob
            x, y = (conf_boxes[i,j,1]+j)/args.S, (conf_boxes[i,j,2]+i)/args.S
            boxes.append(jnp.stack([conf, x, y, conf_boxes[i,j,3], conf_boxes[i,j,4], pred_class]))
    return jnp.array(boxes)

def get_nms_boxes_mAP_coco_mAP(cells, target):
    boxes = get_best_boxes_and_classes(cells)  # Nx6
    boxes = nms(boxes, iou_threshold=0.5, conf_threshold=0.4)  # N'x6
    target_boxes = jnp.concatenate([target[...,20:],jnp.argmax(target[...,:20],-1,keepdims=True)],-1).reshape(-1,6)  # Nx6
    _mAP = mAP(boxes, target_boxes, iou_threshold=0.5)
    _coco_mAP = coco_mAP(boxes, target_boxes)
    return boxes, _mAP, _coco_mAP

if __name__ == '__main__':
    args, writer = get_args_and_writer()
    key = jax.random.PRNGKey(args.seed)
    model = YoloV1()
    print(model.tabulate(key, jnp.empty(args.input_shape), train=False))

    variables = model.init(key, jnp.empty(args.input_shape), train=False)
    state = TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optax.adam(learning_rate=args.learning_rate),
        batch_stats=variables['batch_stats'],
        dropout_key=key
    )

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
    
    from katacv.utils.VOC.build_dataset import VOCBuilder
    ds_builder = VOCBuilder(args)
    train_ds, train_ds_size = ds_builder.get_dataset('train')
    val_ds, val_ds_size = ds_builder.get_dataset('val')

    import time
    from tqdm import tqdm

    start_time, global_step = time.time(), 0
    for epoch in range(1,args.total_epochs+1):
        print(f"epoch: {epoch}/{args.total_epochs}")
        logs.reset()
        print("training...")
        for x, y in tqdm(train_ds, total=train_ds_size, desc="Processing"):
        # for x, y in tqdm(train_ds.take(300), total=300, desc="Processing"):
            x, y = x.numpy(), y.numpy()
            global_step += 1
            print("data shape:", x.shape, y.shape)
            state, (*metrics, cells) = model_step(state, x, y)
            boxes, *mAP_coco_mAP = get_nms_boxes_mAP_coco_mAP(cells, y)
            logs.update(
                ['loss_train', 'loss_coord_train', 'loss_conf_train', 'loss_noobj_train', 'loss_class_train', 'mAP_train', 'coco_mAP_train'],
                [*metrics, *mAP_coco_mAP]
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
        for x, y in tqdm(val_ds, total=val_ds_size, desc="Processing"):
            x, y = x.numpy(), y.numpy()
            state, (*metrics, cells) = model_step(state, x, y)
            boxes, *mAP_coco_mAP = get_nms_boxes_mAP_coco_mAP(cells, y)
            logs.update(
                ['loss_val', 'loss_coord_val', 'loss_conf_val', 'loss_noobj_val', 'loss_class_val', 'mAP_val', 'coco_mAP_val'],
                [*metrics, *mAP_coco_mAP]
            )
        logs.writer_tensorboard(writer, global_step)

        if epoch % args.save_weights_freq == 0:
            path = args.path_cp.joinpath(f"{args.model_name}-{save_id:04}")
            with open(path, 'wb') as file:
                file.write(flax.serialization.to_bytes(state))
            print(f"save weights at '{str(path)}'")
            save_id += 1

    writer.close()