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
2023/12/5: Update image normalized.
'''
import tensorflow as tf
tf.config.experimental.set_visible_devices([], "GPU")  # don't use gpu
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
        area_range=(0.08, 1.0),
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
        img = tf.image.random_flip_left_right(img)
        img /= 255.0
        label = tf.one_hot(label, depth=1000)
        return img, label
    return thunk

from pathlib import Path
DATASET_SIZE = {
    'train': 1281167,
    'val': 50000
}
class ImagenetBuilder:
    
    def __init__(self, args: NamedTuple):
        self.args = args
        self.path = Path(str(args.path_dataset_tfrecord))

    def get_dataset(self, sub_dataset='train'):
        path = self.path.joinpath(f"imagenet2012-{sub_dataset}.tfrecord")
        assert(path.exists())
        ds_tfrecord = tf.data.TFRecordDataset(str(path))
        ds = ds_tfrecord.map(decode_and_aug(
            True if sub_dataset == 'train' else False,
            self.args.image_size,
            self.args.image_center_crop_padding_size
        )).ignore_errors(log_warning=True).shuffle(self.args.shuffle_size).batch(self.args.batch_size, drop_remainder=True)
        ds_size = DATASET_SIZE[sub_dataset] // self.args.batch_size
        return ds, ds_size

if __name__ == '__main__':
    print("Sample from the tfrecord.")
    import argparse
    from pathlib import Path
    cvt2Path = lambda x: Path(x)
    parser = argparse.ArgumentParser()
    # parser.add_argument("--path-dataset-tfrecord", type=cvt2Path, default=Path("/media/yy/Data/dataset/imagenet/"),
    parser.add_argument("--path-dataset-tfrecord", type=cvt2Path, default=Path("/home/wty/Coding/datasets/imagenet/tfrecord"),
        help="the path of the tfrecord dataset directory")
    parser.add_argument("--image-size", type=int, default=224,
        help="the input image size of the model")
    parser.add_argument("--image-center-crop-padding-size", type=int, default=32,
        help="the padding size of the center crop of the origin image")
    parser.add_argument("--batch-size", type=int, default=32,
        help="the size of each batch")
    parser.add_argument("--shuffle-size", type=int, default=32*16,
        help="the shuffle size of the dataset")
    parser.add_argument("--sub-dataset", type=str, default='train',
        help="the sub-dataset of the dataset (train/val)")
    args = parser.parse_args()

    ds_builder = ImagenetBuilder(args)
    ds, ds_size = ds_builder.get_dataset(sub_dataset=args.sub_dataset)
    print("dataset size:", ds_size)

    from label2readable import label2readable

    import matplotlib.pyplot as plt
    import time
    start_time = time.time()
    for img, label in ds.take(5):
        print(tf.reduce_max(img), tf.reduce_min(img))
        plt.imshow(img[0])
        plt.title(label2readable[int(tf.argmax(label[0]))])
        plt.show()
    print("used time:", time.time() - start_time)