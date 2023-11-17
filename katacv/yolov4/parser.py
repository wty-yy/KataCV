from katacv.utils.related_pkgs.utility import *
import katacv.yolov4.config as cfg
from katacv.utils.parser import Parser, CVArgs, cvt2Path, SummaryWriter, datetime, str2bool

class YOLOv4Args(CVArgs):
  ### Dataset ###
  num_classes: int
  use_mosaic4: bool
  num_data_workers: int
  ### Model ###
  anchors: List[Tuple[int, int]]
  path_darknet_weights: Path
  ### Training ###
  warmup_epochs: int
  steps_per_epoch: int
  learning_rate_fn: Callable

def get_args_and_writer(no_writer=False, input_args=None) -> Tuple[YOLOv4Args, SummaryWriter] | YOLOv4Args:
  parser = Parser(model_name="YOLOv4", wandb_project_name=cfg.dataset_name)
  ### Model ###
  parser.add_argument("--anchors", nargs='+', default=cfg.anchors,
    help="the anchors bounding boxes")
  parser.add_argument("--path-darknet-weights", type=cvt2Path, default=cfg.path_darknet_weights,
    help="the path of the CSP-DarkNet53 weights")
  ### Dataset ###
  parser.add_argument("--path-dataset", type=cvt2Path, default=cfg.path_dataset,
    help="the path of the dataset")
  parser.add_argument("--image-shape", nargs='+', default=cfg.image_shape,
    help="the input shape of the YOLOv4 model")
  parser.add_argument("--num-classes", type=int, default=cfg.num_classes,
    help="the number of the classes in dataset")
  parser.add_argument("--use-mosaic4", type=str2bool, default=cfg.use_mosaic4,
    help="if taggled, using mosaic 2x2 in data augmentation")
  parser.add_argument("--num-data-workers", type=int, default=cfg.num_data_workers,
    help="the number of the subprocesses to use for data loading.")
  ### Training ###
  parser.add_argument("--total-epochs", type=int, default=cfg.total_epochs,
    help="the total epochs for training")
  parser.add_argument("--batch-size", type=int, default=cfg.batch_size,
    help="the batch size for training")
  parser.add_argument("--learning-rate", type=float, default=cfg.learning_rate,
    help="the learning rate for training")
  parser.add_argument("--weight-decay", type=float, default=cfg.weight_decay,
    help="the coef of the l2 weight penalty")
  parser.add_argument("--warmup-epochs", type=int, default=cfg.warmup_epochs,
    help="the epochs for warming up the learning rate")
  parser.add_argument("--momentum", type=float, default=cfg.momentum,
    help="the momentum for SGD optimizer")
  args = parser.get_args(input_args)
  args.run_name = f"{args.model_name}__load_{args.load_id}__warmup_lr_{args.learning_rate}__batch_{args.batch_size}__{datetime.datetime.now().strftime(r'%Y%m%d_%H%M%S')}"
  args.steps_per_epoch = cfg.train_ds_size // args.batch_size
  args.input_shape = (args.batch_size, *args.image_shape)
  if no_writer: return args
  
  writer = parser.get_writer(args)
  return args, writer

