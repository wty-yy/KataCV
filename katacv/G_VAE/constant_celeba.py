from pathlib import Path

### Dataset config ###
path_dataset = Path("/home/yy/Coding/datasets/celeba")
batch_size = 32
shuffle_size = 32 * 16
image_shape = (208, 176, 3)
repeat = 1
use_aug = True
train_data_size = 162079 * repeat

### Model config ###
class_num = 4
# encoder_stage_size = (2, 2, 4, 4)
encoder_stage_size = (3, 6, 8, 3)
decoder_stage_size = encoder_stage_size[::-1]
feature_size = 128

### Training config ###
total_epochs = 10
learning_rate = 0.001
coef_kl_loss = 2.5e-3
coef_cls_loss = 5.0
flag_l2_image_loss = True
flag_cosine_schedule = True
