from pathlib import Path

### Dataset config ###
path_dataset = Path("/home/yy/Coding/datasets/celeba")
batch_size = 128
shuffle_size = 128 * 16
# image_shape = (208, 176, 3)
# image_shape = (192, 160, 3)
image_shape = (96, 64, 3)
repeat = 1
use_aug = True
train_data_size = 162079 * repeat

### Model config ###
class_num = 4
# encoder_stage_size = (2, 2, 4, 4)
# encoder_stage_size = (3, 6, 8, 3)
encoder_start_filters = 64
encoder_stage_size = (1, 2, 4, 2)
# decoder_start_filters = encoder_start_filters * (2 ** (len(encoder_stage_size) - 1))  # 512
decoder_start_filters = 1024
# decoder_stage_size = (2, 2, 2)
feature_size = 2048
concat_num = 0

### Training config ###
total_epochs = 10
learning_rate = 5e-4
coef_kl_loss = 2.5e-3
coef_cls_loss = 20.0
flag_l2_image_loss = True
flag_cosine_schedule = True
