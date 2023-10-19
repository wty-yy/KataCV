# -*- coding: utf-8 -*-
'''
@File    : build_dataset.py
@Time    : 2023/10/14 09:45:02
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : 构建OCR所需的数据集
基于MJSynth数据集: https://www.robots.ox.ac.uk/~vgg/data/text/#sec-synth，该数据集共包含8919273个样本，
其中`annotation_train.txt`, `annotation_val.txt`给出了对应的图片地址和其对应的文本，格式如下：

directory1 text_label1
directory2 text_label2
directory3 text_label3
...

下面将对转化好的tfrecord进行读取，需要设定该数据集中图像缩放后的长宽(W,H)，最长文本长度N，全体字符集大小C，及其对应的编码ch2idx，
数据集的格式为 x=(B,H,W), y=(B,N)，空位用空字符进行填充。
'''
from pathlib import Path
from typing import NamedTuple
import tensorflow as tf

def decode_example(target_size, N, ch2idx: tf.lookup.StaticVocabularyTable, use_aug, use_lower):
    def thunk(example):
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.string),
        }
        example = tf.io.parse_single_example(example, feature_description)
        image = tf.io.decode_jpeg(example['image'], channels=1)
        label = tf.strings.lower(example['label']) if use_lower else example['label']
        label = tf.cast(tf.strings.unicode_decode(label, 'UTF-8'), tf.int64)

        image = tf.image.resize(image, target_size)
        label = ch2idx.lookup(label)
        label = tf.concat([label, tf.zeros((N-tf.shape(label)[0],), tf.int64)], 0)

        image = tf.cast(image, tf.uint8)
        return image, label
    return thunk

DATASET_SIZE = {
    'mjsynth': {
        'train': 7224586, # origin: 7224612, destory: 26
        'val': 802731, # origin: 802734, destory: 3
        '8examples': 8,
    }
}
class DatasetBuilder():
    name: str  # mjsynth, auto detecte from `path_dataset_tfrecord`
    path_dataset_tfrecord: Path  # `name-train.tfrecord`, `name-val.tfrecord`, ...
    batch_size: int
    shuffle_size: int
    image_size: tuple
    N: int  # the maximum length of labels
    ch2idx: dict  # the dictionary from characters to indexes
    
    def __init__(self, args: NamedTuple):
        self.path_dataset_tfrecord = args.path_dataset_tfrecord
        self.name = str(self.path_dataset_tfrecord.parent.name)
        self.batch_size, self.shuffle_size = args.batch_size, args.shuffle_size
        self.image_size = (args.image_height, args.image_width)
        self.N = args.max_label_length
        self.ch2idx = tf.lookup.StaticVocabularyTable(
            tf.lookup.KeyValueTensorInitializer(
                list(args.ch2idx.keys()),
                list(args.ch2idx.values()),
                key_dtype=tf.int64,
                value_dtype=tf.int64,
            ),
            num_oov_buckets=1
        )
    
    def get_dataset(self, subset='train', repeat=1, shuffle=True, use_aug=True, use_lower=False):
        ds_tfrecord = tf.data.TFRecordDataset(str(self.path_dataset_tfrecord.joinpath(f"{self.name}-{subset}.tfrecord")))
        ds = ds_tfrecord.map(decode_example(self.image_size, self.N, self.ch2idx, use_aug, use_lower)).repeat(repeat)
        if shuffle: ds = ds.shuffle(self.shuffle_size)
        ds = ds.batch(self.batch_size, drop_remainder=True)
        return ds, DATASET_SIZE[self.name][subset] * repeat // self.batch_size

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    cvt2Path = lambda x: Path(x)
    parser.add_argument("--path-dataset-tfrecord", type=cvt2Path, default=Path("/home/wty/Coding/datasets/mjsynth/tfrecord"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--shuffle-size", type=int, default=1*16)
    parser.add_argument("--image-width", type=int, default=100)
    parser.add_argument("--image-height", type=int, default=32)
    parser.add_argument("--path-lexicon", type=cvt2Path, default=Path("/home/wty/Coding/datasets/mjsynth/lexicon.txt"))
    args = parser.parse_args()

    from check_lexicon import get_info_from_lexicon
    args.max_label_length, args.class_num, args.ch2idx, args.idx2ch = get_info_from_lexicon(args.path_lexicon)
    ds_builder = DatasetBuilder(args)
    # ds, ds_size = ds_builder.get_dataset("train")
    ds, ds_size = ds_builder.get_dataset("val", use_aug=False)
    print("Datasize:", ds_size)
    import matplotlib.pyplot as plt
    from tqdm import tqdm
    total = 0
    for x, y in tqdm(ds, total=ds_size):
        total += 1
        # image = x[0].numpy()
        # label = ''.join([chr(args.idx2ch[i]) for i in y[0].numpy() if i != 0])
        # plt.imshow(image)
        # plt.title(label)
        # plt.show()
    print("total:", total)

 