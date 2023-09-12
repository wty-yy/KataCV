"""
@Author: wty-yy
@Date: 2022-12-24 15:08:30
@LastEditTime: 2022-12-25 11:12:43
@Description: Lenet-5处理mnist数据集
参考文章: https://blog.csdn.net/qq_42570457/article/details/81460807
"""
import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
keras = tf.keras
layers = keras.layers

batch_size = 32
epochs=20

# 1. 数据集准备
(train_x, train_y), (val_x, val_y) = keras.datasets.mnist.load_data()
train_x = tf.expand_dims(tf.constant(train_x, tf.float32), -1) / 255.
val_x = tf.expand_dims(tf.constant(val_x, tf.float32), -1) / 255.
train_ds = tf.data.Dataset.from_tensor_slices((train_x, train_y))
train_ds = train_ds.shuffle(1000).batch(batch_size)
val_ds = tf.data.Dataset.from_tensor_slices((val_x, val_y))
val_ds = val_ds.batch(batch_size)
print(f"图像大小 {train_x[0].shape}, 训练集大小 {train_ds.cardinality()}, 验证集大小 {val_ds.cardinality()}")
tf.sigmoid
# 2. 搭建lenet-5网络
class SigmoidLayer(layers.Layer):
    def __init__(self, name=None):
        super().__init__(name=name)
    
    def call(self, inputs):
        return keras.activations.sigmoid(inputs)

def lenet_5():
    model = keras.Sequential([
        layers.Resizing(32, 32, name='Input'),
        layers.Conv2D(6, kernel_size=5, name='Cov1'),  # 卷积层，28x28x6
        layers.MaxPool2D(2, strides=2, name='Pool1'),   # 汇聚层，14x14x6
        SigmoidLayer(name='Sigmoid1'),  # sigmoid激活函数
        layers.Conv2D(16, kernel_size=5, name='Cov2'), # 卷积层，10x10x6
        layers.MaxPool2D(2, strides=2, name='Pool2'),   # 汇聚层，5x5x16
        SigmoidLayer(name='Sigmoid2'),  # sigmoid激活函数
        layers.Conv2D(120, kernel_size=5, name='Cov3'),# 卷积层，1x1x120

        layers.Flatten(),
        layers.Dense(84, activation='sigmoid', name='Dense'), # 全连接，84
        layers.Dense(10, name='Output')
    ])
    return model

lenet = lenet_5()
lenet.build(input_shape=(None, 28, 28, 1))
lenet.summary()

# 3. 模型训练
batch_N = train_ds.cardinality()
optimizer = keras.optimizers.SGD(learning_rate=0.01)
acc_meter = keras.metrics.SparseCategoricalAccuracy(name='acc')  # 离散准确率计数器
loss_meter = keras.metrics.Mean(name='loss')  # 平均损失值
metrics = [acc_meter, loss_meter]
history = {'acc':[], 'val_acc':[], 'loss':[], 'val_loss':[]}

def plot_figure():
    fig = plt.figure(figsize=(8, 6))
    plt.plot(history['acc'], label='acc')
    plt.plot(history['val_acc'], label='val_acc')
    plt.legend()
    plt.title('Accuracy')
    fig.tight_layout()
    plt.savefig('acc.png')
    plt.close()

def validate(val_ds):
    for step, (x, y) in enumerate(train_ds):
        out = lenet(x, training=False)
        y_onehot = tf.one_hot(y, depth=10)
        loss = tf.reduce_sum(tf.square(out - y_onehot)) / batch_size
        acc_meter.update_state(y, out)
        loss_meter.update_state(loss)

def train():
    for step, (x, y) in tqdm(enumerate(train_ds)):
        with tf.GradientTape() as tape:
            out = lenet(x, training=True)
            y_onehot = tf.one_hot(y, depth=10)
            loss = tf.reduce_sum(tf.square(out - y_onehot)) / batch_size
        grads = tape.gradient(loss, lenet.trainable_variables)
        optimizer.apply_gradients(zip(grads, lenet.trainable_variables))
        acc_meter.update_state(y, out)
        loss_meter.update_state(loss)
        
        if step % 1000 == 0:
            s = f"{step}/{batch_N} "
            for metric in metrics:
                s += f"{metric.name:}={metric.result().numpy():.3f} "
                history[metric.name].append(metric.result())
                metric.reset_states()
            validate(val_ds.take(100))
            for metric in metrics:
                s += f"{'val_'+metric.name:}={metric.result().numpy():.3f} "
                history['val_'+metric.name].append(metric.result())
                metric.reset_states()
            print(s)
            plot_figure()

for epoch in range(epochs):
    print(f"epochs: {epoch+1}")
    train()

print('全部验证集上进行验证')
validate(val_ds)
