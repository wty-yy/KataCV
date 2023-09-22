# -*- coding: utf-8 -*-
'''
@File    : build_dataset.py
@Time    : 2023/09/12 12:21:29
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : 
COCO数据集下载：
'''
from pathlib import Path
from typing import NamedTuple
import tensorflow as tf

def resize_and_augmentation(image: tf.Tensor, params: tf.Tensor, target_size: int, use_aug: bool = True):
    image = tf.image.resize([image], [target_size, target_size])[0]
    flip_rand = tf.cast(tf.random.uniform([]) < 0.5, tf.bool)
    if not use_aug: flip_rand = tf.constant(True)
    def flip_left_right():
        flip_image = tf.image.flip_left_right(image)
        flip_params = tf.concat([1.0 - params[:,0:1], params[:,1:]], axis=1)
        return flip_image, flip_params
    image, params = tf.cond(
        flip_rand,
        lambda: (image, params),
        flip_left_right
    )
    image = tf.cast(image, tf.uint8)
    return image, params

def boxes_to_cells(labels: tf.Tensor, params: tf.Tensor, S: int):
    label = tf.zeros((S, S, 25), tf.float32)
    for o in range(tf.shape(params)[0]):
        j, i = tf.cast(params[o,0]*S, tf.int32), tf.cast(params[o,1]*S, tf.int32)
        one_hot = tf.one_hot(labels[o], 20)
        x_cell = params[o,0]*S - tf.cast(j, tf.float32)
        y_cell = params[o,1]*S - tf.cast(i, tf.float32)
        features = tf.concat([one_hot, [1.0, x_cell, y_cell, params[o,2], params[o,3]]], axis=0)
        label = tf.tensor_scatter_nd_update(label, [(i,j,k) for k in range(25)], features)
    return label

def decode_example(target_size, S, use_aug):
    def thunk(example):
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'labels': tf.io.VarLenFeature(tf.int64),
            'params': tf.io.VarLenFeature(tf.float32)
        }
        example = tf.io.parse_single_example(example, feature_description)
        image = tf.io.decode_jpeg(example['image'], channels=3)
        labels = tf.sparse.to_dense(example['labels'], default_value=0)
        params = tf.reshape(tf.sparse.to_dense(example['params'], default_value=0.0), [-1, 4])
        image, params = resize_and_augmentation(image, params, target_size, use_aug)
        label = boxes_to_cells(labels, params, S)
        return image, label
    return thunk

DATA_SIZE = {
    'train': 16551,
    'val': 4952,
    '8examples': 8,
    '100examples': 103,
}
class COCOBuilder():
    path_dataset_tfrecord: Path  # `COCO-train.tfrecord`, `VOC-val.tfrecord`, ...
    batch_size: int
    shuffle_size: int
    image_size: int
    S: int  # the split size of the cells, divide the figure to SxS cells
    
    def __init__(self, args: NamedTuple):
        self.path_dataset_tfrecord = args.path_dataset_tfrecord
        self.batch_size, self.shuffle_size = args.batch_size, args.shuffle_size
        self.image_size = args.image_size
        self.S = args.split_size
    
    def get_dataset(self, subset='train', repeat=1, shuffle=True, use_aug=True):
        ds_tfrecord = tf.data.TFRecordDataset(str(self.path_dataset_tfrecord.joinpath(f"COCO-{subset}.tfrecord")))
        ds = ds_tfrecord.map(decode_example(self.image_size, self.S, use_aug)).repeat(repeat)
        if shuffle: ds = ds.shuffle(self.shuffle_size)
        ds = ds.batch(self.batch_size, drop_remainder=True)
        return ds, DATA_SIZE[subset] * repeat // self.batch_size

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    cvt2Path = lambda x: Path(x)
    parser.add_argument("--path-dataset-tfrecord", type=cvt2Path, default=Path("/home/yy/Coding/datasets/COCO/tfrecord"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--shuffle-size", type=int, default=64*16)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--split-size", type=int, default=7)
    args = parser.parse_args()

    ds_builder = COCOBuilder(args)
    # ds, ds_size = ds_builder.get_dataset("train")
    ds, ds_size = ds_builder.get_dataset("val", use_aug=False)
    print("Datasize:", ds_size)

    from katacv.utils.detection import plot_box, plot_cells
    import time, numpy as np
    start_time = time.time()
    import matplotlib.pyplot as plt
    from label2realname import label2realname
    from tqdm import tqdm

    S = args.split_size
    for image, label in tqdm(ds):
        image, label = image[0], label[0]
        # continue
        # TODO: 验证label是否正确 OK
        fig, ax = plt.subplots()
        ax.imshow(image)
        label = label.numpy()
        for i in range(S):
            for j in range(S):
                if label[i,j,20] == 1:
                    cat = np.argmax(label[i,j,:20])
                    features = label[i,j]
                    x, y = (features[21]+j)/S, (features[22]+i)/S
                    text = f"{label2realname[cat]} ({i},{j})"
                    plot_box(ax, image.shape, (x, y, features[23], features[24]), text)
        plot_cells(ax, image.shape, S)
        plt.savefig("test_build_dataset.jpg", dpi=200)
        plt.show()
    print("used time:", time.time() - start_time)
