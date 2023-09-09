# -*- coding: utf-8 -*-
'''
@File    : build_dataset.py
@Time    : 2023/09/08 20:50:16
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : 

DatasetBuilder用于生成tf.data.Dataset训练集，该数据的输出保证：
1. 训练集 ds_build.get_dataset(train=True)：
   原图像先随机裁剪为近似正方形，裁剪后面积占比范围(0.7, 1.0)，若随机裁剪失败，则使用测试集的中心裁剪
   测试集 ds_build.get_dataset(train=False)：
   直接按照中心进行裁剪，按照目标尺寸预留padding边界大小，按照图像最小边等比例裁剪
2. 再缩放成固定大小(image_size,image_size,3)，uint8数据类型，对label展开成one-hot向量输出
要求传入的args至少包含以下三个参数

args = Args(
    path_dataset_tfrecord = "../datasets/imagenet/imagenet2012-train.tfrecord",
    image_size = 224,
    image_center_crop_padding_size = 32,
    batch_size=128,
    shuffle_size=128*16
)
'''
import tensorflow as tf
from typing import NamedTuple

def preprocess_tfrecord(example):
    feature = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, feature)
    return example['image'], example['label']

def get_random_crop(bytes, shape):
    bbox = tf.image.sample_distorted_bounding_box(
        shape,  # should be slow, since shape is change, so the graph is change
        bounding_boxes=tf.reshape([0.0, 0.0, 1.0, 1.0], shape=[1, 1, 4]),
        min_object_covered=0.1,
        aspect_ratio_range=(3/4, 4/3),
        area_range=(0.7, 1.0),
        max_attempts=10
    )
    offset = bbox.begin[:2]
    size = bbox.size[:2]
    return tf.io.decode_and_crop_jpeg(bytes, tf.concat([offset, size], axis=0), channels=3)

def get_center_crop(bytes, shape, padding, target_size):
    crop_size = tf.cast(
        target_size / (target_size + padding) * tf.cast(
            tf.minimum(shape[0], shape[1]), tf.float32
        ), tf.int32
    )
    offset = [(shape[0]-crop_size+1) // 2, (shape[1]-crop_size+1) // 2]
    return tf.io.decode_and_crop_jpeg(bytes, tf.stack(offset + [crop_size]*2), channels=3)        

def decode_and_aug(train: bool, target_size, padding):
    def thunk(example):
        feature = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        }
        example = tf.io.parse_single_example(example, feature)
        bytes, label = example['image'], example['label']
        origin_shape = tf.io.extract_jpeg_shape(bytes)
        if train:
            img = get_random_crop(bytes, origin_shape)
            img = tf.cond(  # if random crop failed, the return shape will be origin_shape
                tf.equal(tf.reduce_sum(tf.cast(tf.equal(origin_shape, tf.shape(img)), tf.int32)), 3),
                lambda: get_center_crop(bytes, origin_shape, padding, target_size),
                lambda: img
            )
        else: img = get_center_crop(bytes, origin_shape, padding, target_size)
        img = tf.image.resize([img], [target_size, target_size])[0]
        img = tf.cast(img, tf.uint8)
        img = tf.image.random_flip_left_right(img)
        label = tf.one_hot(label, depth=1000)
        return img, label
    return thunk

class DatasetBuilder:
    
    def __init__(self, args: NamedTuple):
        self.args = args
        self.ds_tfrecord = tf.data.TFRecordDataset(args.path_dataset_tfrecord)

    def get_dataset(self, train=True):
        ds_tfrecord = tf.data.TFRecordDataset(self.args.path_dataset_tfrecord)
        return ds_tfrecord.map(decode_and_aug(
            train,
            self.args.image_size,
            self.args.image_center_crop_padding_size
        )).shuffle(args.shuffle_size).batch(args.batch_size)

if __name__ == '__main__':
    class Args(NamedTuple):
        path_dataset_tfrecord: str
        image_size: int
        image_center_crop_padding_size: int
        batch_size: int
        shuffle_size: int

    # from pathlib import Path
    args = Args(
        path_dataset_tfrecord = "/media/yy/Data/dataset/imagenet/imagenet2012-val.tfrecord",
        image_size = 224,
        image_center_crop_padding_size = 32,
        batch_size=128,
        shuffle_size=128*16
    )
    ds_builder = DatasetBuilder(args)
    ds = ds_builder.get_dataset(train=False)

    from label2readable import label2readable

    import matplotlib.pyplot as plt
    import time
    start_time = time.time()
    for img, label in ds.take(5):
        plt.imshow(img[0])
        plt.title(label2readable[int(label[0])])
        plt.show()
    print("used time:", time.time() - start_time)