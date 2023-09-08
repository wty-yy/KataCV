import tensorflow as tf
import matplotlib.pyplot as plt
from tqdm import tqdm
from constant import PATH_DATASET, PATH_HISTORY, PATH_WEIGHT, load_weight_num, PATH_LOAD_WEIGHT
import json
from datetime import datetime
json.encoder.FLOAT_REPR = lambda x: format(x, '.2f')
keras = tf.keras
layers = keras.layers
BATCH_SIZE = 64

ds_train = tf.keras.utils.image_dataset_from_directory(PATH_DATASET.joinpath('train'), batch_size=BATCH_SIZE)
ds_val = tf.keras.utils.image_dataset_from_directory(PATH_DATASET.joinpath('val'), batch_size=32)

def conver_data(x, y):
    x = x / 255.
    y = tf.one_hot(y, depth=1000)
    return x, y

tf.keras.mixed_precision.set_global_policy('mixed_float16')

# ds_train = ds_train.map(conver_data, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
# ds_val = ds_val.map(conver_data, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
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
    x = layers.Dense(4096, activation='relu', name='Dense1', dtype='float16')(x)
    x = layers.Dense(4096, activation='relu', name='Dense2', dtype='float16')(x)
    outputs = layers.Dense(1000, activation='softmax', dtype='float16', name='Output')(x)
    return keras.Model(inputs, outputs)
vgg16 = build_model()
keras.utils.plot_model(vgg16, show_shapes=True, to_file="VGG16.png")
vgg16.summary()

if PATH_LOAD_WEIGHT is not None:
    vgg16.load_weights(PATH_LOAD_WEIGHT)
    print(f"Load weight from: {PATH_LOAD_WEIGHT}")

vgg16.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5), loss=keras.losses.CategoricalCrossentropy(from_logits=False),
                metrics=[keras.metrics.TopKCategoricalAccuracy(1, name="Top1"), keras.metrics.TopKCategoricalAccuracy(5, name="Top5")])

optimizer = keras.optimizers.Adam(learning_rate=1e-5)
loss_fn = keras.losses.CategoricalCrossentropy(from_logits=False)
metrics = [keras.metrics.Mean(name='loss'),
           keras.metrics.TopKCategoricalAccuracy(1, name="Top1"),
           keras.metrics.TopKCategoricalAccuracy(5, name="Top5")]
history_train = {"loss": [], "Top1": [], "Top5": []}
history_val = {"loss": [], "Top1": [], "Top5": []}
start_time = datetime.now().strftime("%Y%m%d-%H%M%S")
print(f"Start training at {start_time}")

def update_metrics(loss, y, outputs):
    metrics[0].update_state(loss)
    for i in range(1, 3):
        metrics[i].update_state(y, outputs)

@tf.function
def validate_step(X, y):
    outputs = vgg16(X, training=False)
    loss = loss_fn(y, outputs)
    update_metrics(loss, y, outputs)

def validate(val_ds):
    for X, y in val_ds:
        validate_step(X, y)

def reset_metrics(history):
    message = ""
    for metric in metrics:
        message += f"{metric.name}={metric.result().numpy():.3f} "
        history[metric.name].append(round(float(metric.result()), 5))
        metric.reset_states()
    return message

def history_to_json(history, fname):
    path = PATH_HISTORY.joinpath(fname+'-'+start_time+".json")
    print(f"save {fname} history at '{path}'")
    with open(path, 'w') as file:
        json.dump(history, file, indent=2)

def save_weights(model, epoch):
    path = PATH_WEIGHT.joinpath(f"cp-{load_weight_num+1+epoch:04}")
    model.save_weights(path)
    print(f"save weights at '{path}'")

train_N = ds_train.cardinality().numpy()
val_N = ds_val.cardinality().numpy()
print(f"{train_N=}, {val_N=}")

@tf.function
def train_step(X, y):
    with tf.GradientTape() as tape:
        outputs = vgg16(X, training=True)
        loss = loss_fn(y, outputs)
    grads = tape.gradient(loss, vgg16.trainable_variables)
    optimizer.apply_gradients(zip(grads, vgg16.trainable_variables))
    return loss, outputs

def train():
    for epoch in range(30):
        print(f"epoch: {epoch+1}/30")
        step = 0
        for X, y in tqdm(ds_train):
            step += 1
            loss, outputs = train_step(X, y)
            update_metrics(loss, y, outputs)
            if step % 3000 == 0 or step == train_N:
                message = f"step: {step}/{train_N}\n"
                message += "train: " + reset_metrics(history_train)
                message += '\n'
                validate(ds_val.take(300))
                message += "val: " + reset_metrics(history_val)
                print(message)
                # keras.backend.clear_session()  # It's bad

        print(f"{history_train=}")
        print(f"{history_val=}")

        save_weights(vgg16, epoch)

        history_to_json(history_train, 'train')
        history_to_json(history_val, 'val')
train()

# vgg16.fit(ds_train, epochs=30, validation_data=ds_val, callbacks=[tensorboard_callback, cp_callback])
# vgg16.fit(ds_train, epochs=30, callbacks=[tensorboard_callback, cp_callback])
