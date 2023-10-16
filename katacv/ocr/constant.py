from katacv.utils.related_pkgs.utility import *

### Dataset ###
path_dataset_tfrecord = Path("/home/wty/Coding/datasets/mjsynth/tfrecord")
batch_size = 128
shuffle_size = batch_size * 16
image_width = 100
image_height = 32
max_label_length = 23
character_set = {
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
    'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'
}

from katacv.utils.ocr.build_dataset import DATASET_SIZE
train_dataset_size = DATASET_SIZE['mjsynth']['train']
steps_per_epoch = train_dataset_size // batch_size

### Training ###
total_epochs = 30
learning_rate = 1e-3
weight_decay = 1e-4
warmup_epochs = 5
