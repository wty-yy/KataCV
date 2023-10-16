from katacv.utils.related_pkgs.utility import *
from katacv.utils.parser import Parser, CVArgs, cvt2Path, SummaryWriter, datetime
import katacv.ocr.constant as const

class OCRArgs(CVArgs):
    ### Dataset ###
    class_num: int;
    max_label_length: int;
    ch2idx: dict; idx2ch: dict
    ### Training ###
    warmup_epochs: int
    steps_per_epoch: int
    learning_rate_fn: Callable

def get_args_and_writer(no_writer=False, input_args=None) -> Tuple[OCRArgs, SummaryWriter] | OCRArgs:
    # parser = Parser(model_name="OCR-CNN", wandb_project_name="mjsynth")
    # parser = Parser(model_name="OCR-CRNN-LSTM", wandb_project_name="mjsynth")
    parser = Parser(model_name="OCR-CRNN-BiLSTM", wandb_project_name="mjsynth")
    ### Dataset config ###
    parser.add_argument("--path-dataset-tfrecord", type=cvt2Path, default=const.path_dataset_tfrecord,
        help="the tfrecord directory of the mjsynth dataset")
    parser.add_argument("--batch-size", type=int, default=const.batch_size,
        help="the batch size of the train and validate dataset")
    parser.add_argument("--shuffle-size", type=int, default=const.shuffle_size,
        help="the shuffle size of the tf.data.Dataset")
    parser.add_argument("--image-width", type=int, default=const.image_width,
        help="the width of the input image")
    parser.add_argument("--image-height", type=int, default=const.image_height,
        help="the height of the input image")
    parser.add_argument("--max-label-length", type=int, default=const.max_label_length,
        help="the maximum length of the labels")
    parser.add_argument("--character-set", nargs='+', default=const.character_set,
        help="the character set of the dataset")
    ### Training config ###
    parser.add_argument("--total-epochs", type=int, default=const.total_epochs,
        help="the total epochs for training the model")
    parser.add_argument("--learning-rate", type=float, default=const.learning_rate,
        help="the maximum learning rate of training")
    parser.add_argument("--weight-decay", type=float, default=const.weight_decay,
        help="the coef of l2 weight penalty")
    parser.add_argument("--warmup-epochs", type=float, default=const.weight_decay,
        help="the warming up epochs of cosine learning rate")
    args = parser.get_args(input_args)

    args.character_set = [0] + sorted(ord(c) for c in list(args.character_set))
    args.class_num = len(args.character_set)
    args.ch2idx = {args.character_set[i]: i for i in range(len(args.character_set))}
    args.idx2ch = dict(enumerate(args.character_set))

    # args.run_name = f"{args.model_name}__load_{args.load_id}__warmup_lr_{args.learning_rate}__batch_{args.batch_size}__{datetime.datetime.now().strftime(r'%Y%m%d_%H%M%S')}".replace("/", "-")
    args.write_tensorboard_freq = 100
    args.input_shape = (args.batch_size, args.image_height, args.image_width, 1)
    args.steps_per_epoch = const.steps_per_epoch
    if no_writer: return args

    writer = parser.get_writer(args)
    return args, writer

