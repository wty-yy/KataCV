import sys, os
sys.path.append(os.getcwd())
from katacv.utils.related_pkgs.utility import *
from katacv.utils.related_pkgs.jax_flax_optax_orbax import *
from katacv.yolov3.yolov3_model import NeckModel
from katacv.yolov3.logs import logs
import katacv.yolov3.constant_coco as const

from katacv.utils.parser import Parser, CVArgs, cvt2Path, SummaryWriter
class YOLOv3Args(CVArgs):
    split_size: int;    S: int
    bounding_box: int;  B: int
    class_num: int;     C: int
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
    parser.add_argument("--split-size", type=int, default=const.split_sizes,
        help="the split size of the cells")
    parser.add_argument("--class-num", type=int, default=const.class_num,
        help="the number of the classes (labels)")
    parser.add_argument("--bounding-box", type=int, default=const.bounding_box,
        help="the number of bounding box in each cell (relative to the anchor boxes)")
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
    args.S, args.B, args.C = args.split_size, args.bounding_box, args.class_num
    args.path_darknet = args.path_logs.joinpath(f"DarkNet53-checkpoints/DarkNet53-{args.darknet_id:04}")
    args.input_shape = (args.batch_size, args.image_size, args.image_size, 3)
    if no_writer: return args
    writer = parser.get_writer(args)
    return args, writer

if __name__ == '__main__':
    args, writer = get_args_and_writer()
