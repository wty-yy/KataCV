from katacv.utils.related_pkgs.utility import *
from katacv.utils.related_pkgs.jax_flax_optax_orbax import *
import katacv.yolov3.constant as const
from katacv.utils.parser import Parser, CVArgs, cvt2Path, SummaryWriter, datetime

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
    freeze: bool

def get_args_and_writer(no_writer=False, input_args=None) -> tuple[YOLOv3Args, SummaryWriter] | YOLOv3Args:
    parser = Parser(model_name="YOLOv3-COCO", wandb_project_name="COCO")
    # parser = Parser(model_name="YOLOv3-PASCAL", wandb_project_name="PASCAL VOC")
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
    parser.add_argument("--warmup-epochs", type=int, default=const.warmup_epochs,
        help="the epochs for warming up the learning rate")
    parser.add_argument("--freeze", type=bool, default=const.freeze, const=True, nargs='?',
        help="if targgled, the darnet model will be freezed")
    ### Model config ###
    parser.add_argument("--path-darknet", type=cvt2Path, default=const.path_darknet)
    parser.add_argument("--darknet-id", type=int, default=const.darknet_id,
        help="the weights id of the darknet model")

    args = parser.get_args(input_args)
    assert(args.total_epochs > args.warmup_epochs)
    args.run_name = f"{args.model_name}__load_{args.load_id}__warmup_lr_{args.learning_rate}__batch_{args.batch_size}__freeze_{args.freeze}__{datetime.datetime.now().strftime(r'%Y%m%d_%H%M%S')}".replace("/", "-")
    args.write_tensorboard_freq = 100
    args.B, args.C = args.bounding_box, args.class_num
    args.path_darknet = args.path_darknet.joinpath(f"DarkNet53-{args.darknet_id:04}-lite")
    args.input_shape = (args.batch_size, args.image_size, args.image_size, 3)
    if no_writer: return args

    writer = parser.get_writer(args)
    return args, writer
