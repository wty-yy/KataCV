from pathlib import Path

### Dataset config ###
path_dataset = Path("/home/yy/Coding/datasets/cifar10")
# path_dataset = Path("/home/wty/Coding/datasets/cifar10")
batch_size = 128
shuffle_size = 128 * 16
image_shape = (32, 32, 3)
repeat = 20
use_aug = True
train_data_size = 50000 * repeat

### Model config ###
class_num = 10
encoder_stage_size = (3, 4, 6, 3)
decoder_stage_size = encoder_stage_size[::-1]
feature_size = 512

### Training config ###
total_epochs = 5
learning_rate = 0.001
coef_kl_loss = 2.5e-3
# coef_kl_loss = 1e-4
coef_cls_loss = 5.0
flag_l2_image_loss = True
flag_cosine_schedule = True
