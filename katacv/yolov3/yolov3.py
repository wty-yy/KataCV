import sys, os
sys.path.append(os.getcwd())
from katacv.utils.related_pkgs.utility import *
from katacv.utils.related_pkgs.jax_flax_optax_orbax import *
from katacv.yolov3.logs import logs
import katacv.yolov3.constant_coco as const

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
    parser = Parser(model_name="YOLOv3", wandb_project_name="COCO")
    # COCO Dataset config
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
    parser.add_argument("--coef-noobj", type=float, default=const.coef_noobj,
        help="the coef of the no object loss")
    parser.add_argument("--coef-coord", type=float, default=const.coef_coord,
        help="the coef of the coordinate loss")
    parser.add_argument("--coef-obj", type=float, default=const.coef_obj,
        help="the coef of the object loss")
    parser.add_argument("--coef-class", type=float, default=const.coef_class,
        help="the coef of the class loss")
    # Model config
    parser.add_argument("--total-epochs", type=int, default=40,
        help="the total epochs of the training")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
        help="the coef of the weight penalty")
    parser.add_argument("--darknet-id", type=int, default=50,
        help="the weights id of the darknet model")

    args = parser.get_args()
    args.write_tensorboard_freq = 10
    args.B, args.C = args.bounding_box, args.class_num
    args.path_darknet = args.path_logs.joinpath(f"DarkNet53-checkpoints/DarkNet53-{args.darknet_id:04}-lite")
    args.input_shape = (args.batch_size, args.image_size, args.image_size, 3)
    if no_writer: return args

    writer = parser.get_writer(args)
    return args, writer

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
        state.params['DarkNet_0'] = weights['params']['DarkNet_0']
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
    from katacv.yolov3.yolov3_loss import model_step
    start_time, global_step = time.time(), 0
    if args.train:
        for epoch in range(1, args.total_epochs + 1):
            print(f"epoch: {epoch}/{args.total_epochs}")
            print("training...")
            logs.reset()
            for x, y in tqdm(train_ds, total=train_ds_size):
                x = x.numpy(); y = split_targets(y, args)
                global_step += 1
                state, loss = model_step(state, x, y, train=True, args=args)
                logs.update(['loss_train'], [loss])
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
                _, loss = model_step(state, x, y, train=False, args=args)
                logs.update(['loss_val'], [loss])
            logs.writer_tensorboard(writer, global_step)

            ### Save weights ###
            if epoch % args.save_weights_freq == 0:
                path_save = args.path_cp.joinpath(f"{args.model_name}-{save_id:04}")
                with open(path_save, 'wb') as file:
                    file.write(flax.serialization.to_bytes(state))
                print(f"Save weights at '{str(path_save)}'")
                save_id += 1
    
    writer.close()
