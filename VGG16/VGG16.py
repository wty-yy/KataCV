import tensorflow as tf
import matplotlib.pyplot as plt
keras = tf.keras
layers = keras.layers
BATCH_SIZE = 64

from pathlib import Path
PATH_DATASET = Path("D:\dataset\imagenet")
ds_train = tf.keras.utils.image_dataset_from_directory(PATH_DATASET.joinpath('train'), batch_size=BATCH_SIZE)
ds_val = tf.keras.utils.image_dataset_from_directory(PATH_DATASET.joinpath('val'), batch_size=BATCH_SIZE)

def conver_data(x, y):
    x = x / 255.
    y = tf.one_hot(y, depth=1000)
    return x, y

ds_train = ds_train.map(conver_data, num_parallel_calls=tf.data.AUTOTUNE)
ds_val = ds_val.map(conver_data, num_parallel_calls=tf.data.AUTOTUNE)

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomCrop(224, 224)
], name='Augmentation')

def build_model(inputs_shape=(256,256,3)):
    inputs = layers.Input(shape=inputs_shape, name='img')
    x = data_augmentation(inputs)
    # Block1
    x = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu', name='Conv1')(x)
    x = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu', name='Conv2')(x)
    x = layers.MaxPool2D(2, strides=2, name='Pool1')(x)
    # Block2
    x = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu', name='Conv3')(x)
    x = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu', name='Conv4')(x)
    x = layers.MaxPool2D(2, strides=2, name='Pool2')(x)
    # Block3
    x = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu', name='Conv5')(x)
    x = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu', name='Conv6')(x)
    x = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu', name='Conv7')(x)
    x = layers.MaxPool2D(2, strides=2, name='Pool3')(x)
    # Block4
    x = layers.Conv2D(512, kernel_size=3, padding='same', activation='relu', name='Conv8')(x)
    x = layers.Conv2D(512, kernel_size=3, padding='same', activation='relu', name='Conv9')(x)
    x = layers.Conv2D(512, kernel_size=3, padding='same', activation='relu', name='Conv10')(x)
    x = layers.MaxPool2D(2, strides=2, name='Pool4')(x)
    # Block5
    x = layers.Conv2D(512, kernel_size=3, padding='same', activation='relu', name='Conv11')(x)
    x = layers.Conv2D(512, kernel_size=3, padding='same', activation='relu', name='Conv12')(x)
    x = layers.Conv2D(512, kernel_size=3, padding='same', activation='relu', name='Conv13')(x)
    x = layers.MaxPool2D(2, strides=2, name='Pool5')(x)
    x = layers.Flatten(name='Flatten')(x)
    # FC
    x = layers.Dense(4096, activation='relu', name='Dense1')(x)
    x = layers.Dense(4096, activation='relu', name='Dense2')(x)
    outputs = layers.Dense(1000, activation='softmax')(x)
    return keras.Model(inputs, outputs)
vgg16 = build_model()
keras.utils.plot_model(vgg16, show_shapes=True, to_file="VGG16.png")
vgg16.summary()

vgg16.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-4), loss=keras.losses.CategoricalCrossentropy(from_logits=False),
                metrics=[keras.metrics.TopKCategoricalAccuracy(1, name="Top1"), keras.metrics.TopKCategoricalAccuracy(5, name="Top5")])

import datetime
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# checkpoint_dirpath = Path.cwd().joinpath("training")
# lastest_cp_path = tf.train.latest_checkpoint(checkpoint_dirpath)
# if lastest_cp_path is not None:
#     import re
#     info = re.findall(r"cp-(\d{4})(\+\d*)?", lastest_cp_path)[0]
#     begin_epoch = int(info[0]) + int(0 if info[1] == "" else info[1])
#     alexnet.load_weights(lastest_cp_path)
#     print(f"Load weight from {lastest_cp_path}.")
#     print(f"Start from epoch {begin_epoch}")
# else: begin_epoch = 0
# checkpoint_path = "training/cp-{epoch:04d}+"+str(begin_epoch)+".ckpt"

checkpoint_path = "training_test/cp-{epoch:04d}.ckpt"

# Create a callback that saves the model's weights
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    verbose=1, 
    save_weights_only=True)

vgg16.fit(ds_train, epochs=30, validation_data=ds_val, callbacks=[tensorboard_callback, cp_callback])