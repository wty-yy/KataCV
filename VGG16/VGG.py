import tensorflow as tf
keras = tf.keras
layers = keras.layers

def build_model(input_shape=(224,224,3)):
    inputs = layers.Input(input_shape, name='img')
    # Block1
    x = layers.Conv2D(64, kernel_size=3, padding='same', name='Conv1', activation='relu')(inputs)
    x = layers.Conv2D(64, kernel_size=3, padding='same', name='Conv2', activation='relu')(x)
    x = layers.MaxPool2D(2, strides=2, name='Pool1')(x)
    # Block2
    x = layers.Conv2D(128, kernel_size=3, padding='same', name='Conv3', activation='relu')(x)
    x = layers.Conv2D(128, kernel_size=3, padding='same', name='Conv4', activation='relu')(x)
    x = layers.MaxPool2D(2, strides=2, name='Pool2')(x)
    # Block3
    x = layers.Conv2D(256, kernel_size=3, padding='same', name='Conv5', activation='relu')(x)
    x = layers.Conv2D(256, kernel_size=3, padding='same', name='Conv6', activation='relu')(x)
    x = layers.Conv2D(256, kernel_size=3, padding='same', name='Conv7', activation='relu')(x)
    x = layers.MaxPool2D(2, strides=2, name='Pool3')(x)
    # Block4
    x = layers.Conv2D(512, kernel_size=3, padding='same', name='Conv8', activation='relu')(x)
    x = layers.Conv2D(512, kernel_size=3, padding='same', name='Conv9', activation='relu')(x)
    x = layers.Conv2D(512, kernel_size=3, padding='same', name='Conv10', activation='relu')(x)
    x = layers.MaxPool2D(2, strides=2, name='Pool4')(x)
    # Block5
    x = layers.Conv2D(512, kernel_size=3, padding='same', name='Conv11', activation='relu')(x)
    x = layers.Conv2D(512, kernel_size=3, padding='same', name='Conv12', activation='relu')(x)
    x = layers.Conv2D(512, kernel_size=3, padding='same', name='Conv13', activation='relu')(x)
    x = layers.MaxPool2D(2, strides=2, name='Pool5')(x)
    x = layers.Flatten()(x)
    # FC
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dense(4096, activation='relu')(x)
    x = layers.Dense(1000, activation='softmax')(x)
    return keras.Model(inputs, x)

vgg16 = build_model()
vgg16.summary()