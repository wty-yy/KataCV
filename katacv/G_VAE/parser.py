from katacv.utils.related_pkgs.utility import *
from katacv.utils.parser import Parser, cvt2Path, SummaryWriter, CVArgs, datetime
import katacv.G_VAE.constant_mnist as const_mnist
import katacv.G_VAE.constant_cifar10 as const_cifar10
import katacv.G_VAE.constant_celeba as const_celeba
import katacv.G_VAE.constant_celeba_unet as const_celeba_unet

dataset2const = {
  'MNIST': const_mnist,
  'cifar10': const_cifar10,
  'celeba': [const_celeba, const_celeba_unet],
}

class VAEArgs(CVArgs):
  ### Model ###
  class_num: int
  encoder_stage_size: Tuple[int]
  decoder_stage_size: Tuple[int]  # no unet
  encoder_start_filters: int  # unet
  decoder_start_filters: int  # unet
  concat_num: int  # uint
  feature_size: int
  use_unet: bool
  ### Train ###
  coef_kl_loss: float
  coef_cls_loss: float
  flag_l2_image_loss: bool
  flag_cosine_schedule: bool
  learning_rate_fn: Callable
  steps_per_epoch: int
  ### Dataset ###
  path_dataset: Path
  repeat: int
  flag_use_aug: bool

def get_args_and_writer(
    no_writer=False, input_args=None,
    model_name="G-VAE", dataset_name="MNIST",
    use_unet=False,
  ) -> Tuple[VAEArgs, SummaryWriter] | VAEArgs:
  assert(dataset_name in dataset2const.keys())
  parser = Parser(model_name=model_name, wandb_project_name=dataset_name)
  const = dataset2const[dataset_name][int(use_unet)]
  ### Dataset config ###
  parser.add_argument("--path-dataset", type=cvt2Path, default=const.path_dataset,
    help="the path of the dataset")
  parser.add_argument("--batch-size", type=int, default=const.batch_size,
    help="the batch size of train and validate dataset")
  parser.add_argument("--shuffle-size", type=int, default=const.shuffle_size,
    help="the shuffle size of the tf.data.Dataset")
  parser.add_argument("--image-shape", default=const.image_shape, nargs='+',
    help="the image shape of the model input")
  parser.add_argument("--repeat", type=int, default=const.repeat,
    help="the number of repeating the train dataset")
  parser.add_argument("--flag-use-aug", type=str2bool, default=const.use_aug, const=True, nargs='?',
    help="if taggled, the augmentation will be used in train dataset")
  ### Model config ###
  parser.add_argument("--class-num", type=int, default=const.class_num,
    help="the number of classification classes")
  parser.add_argument("--encoder-stage-size", nargs="+", default=const.encoder_stage_size,
    help="the encoder stage size of the number of resblock")
  if not use_unet:
    parser.add_argument("--decoder-stage-size", nargs="+", default=const.decoder_stage_size,
      help="the decoder stage size of the number of resblock")
  else:
    parser.add_argument("--encoder-start-filters", type=int, default=const.encoder_start_filters,
      help="the start filters size of the encoder model")
    parser.add_argument("--decoder-start-filters", type=int, default=const.decoder_start_filters,
      help="the start filters size of the decoder model")
    parser.add_argument("--concat-num", type=int, default=const.concat_num,
      help="the number of concat the previous features")
  parser.add_argument("--feature-size", type=int, default=const.feature_size,
    help="the dimension size of the feature")
  ### Training config ###
  parser.add_argument("--total-epochs", type=int, default=const.total_epochs,
    help="the total epochs of training")
  parser.add_argument("--learning-rate", type=float, default=const.learning_rate,
    help="the learning rate of training")
  parser.add_argument("--coef-kl-loss", type=float, default=const.coef_kl_loss,
    help="the coef of the kl loss in VAE")
  parser.add_argument("--coef-cls-loss", type=float, default=const.coef_cls_loss,
    help="the coef of the classification loss in VAE")
  parser.add_argument("--flag-l2-image-loss", type=str2bool, default=const.flag_l2_image_loss, const=True, nargs='?',
    help="if taggled, the l2 image loss will be used")
  parser.add_argument("--flag-cosine-schedule", type=str2bool, default=const.flag_cosine_schedule, const=True, nargs='?',
    help="if taggled, the cosine schedule will be used in learing rate decay")
  args = parser.parse_args(input_args)

  args.model_name += str(args.feature_size)
  args.use_unet = use_unet
  if use_unet: args.model_name += "-UNet"
  parser.check_args(args)
  args.run_name = (
    # f"{args.model_name}_classify__load_{args.load_id}__{'cosine__' if args.flag_cosine_schedule else ''}lr_{args.learning_rate}__"
    f"{args.model_name}__load_{args.load_id}__{'cosine__' if args.flag_cosine_schedule else ''}lr_{args.learning_rate}__"
    f"batch_{args.batch_size}__repeat_{args.repeat}__image_loss_{'l2' if args.flag_l2_image_loss else 'l1'}__"
    f"{datetime.datetime.now().strftime(r'%Y%m%d_%H%M%S')}".replace("/", "-")
  )
  args.input_shape = (args.batch_size, *args.image_shape)
  args.steps_per_epoch = const.train_data_size // args.batch_size
  if no_writer: return args

  writer = parser.get_writer(args)
  return args, writer
  