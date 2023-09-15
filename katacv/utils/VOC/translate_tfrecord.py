# -*- coding: utf-8 -*-
'''
@File    : translate_tfrecord.py
@Time    : 2023/09/12 13:55:31
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : 
将VOC数据集转化为tfrecord
'''
from pathlib import Path
from typing import NamedTuple
import tensorflow as tf, csv

def create_example(path_image, path_label):
    image_bytes = tf.io.read_file(str(path_image))
    image = tf.io.decode_jpeg(image_bytes, channels=3)
    image_bytes = tf.io.encode_jpeg(image)
    
    label_bytes = tf.io.read_file(str(path_label))
    labels = []
    params = []
    for box in tf.strings.split(label_bytes, '\n'):
        parts = tf.strings.strip(tf.strings.split(box, ' '))
        if len(parts) != 5: break
        label = tf.strings.to_number(parts[0], out_type=tf.int64)
        x = tf.strings.to_number(parts[1], out_type=tf.float32)
        y = tf.strings.to_number(parts[2], out_type=tf.float32)
        w = tf.strings.to_number(parts[3], out_type=tf.float32)
        h = tf.strings.to_number(parts[4], out_type=tf.float32)
        labels.append(label)
        params.append([x, y, w, h])
    feature_dict = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes.numpy()])),
        'labels': tf.train.Feature(int64_list=tf.train.Int64List(value=labels)),
        'params': tf.train.Feature(float_list=tf.train.FloatList(value=tf.reshape(params, [-1])))
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example

from tqdm import tqdm

class TFrecordTranslater:
    path_images: Path  # 1.jpg, 2.jpg, ...
    path_labels: Path  # 1.txt, 2.txt, ...
    path_tfrecord: Path  # save the tfrecord
    path_idxs: Path  # csv

    def __init__(self, args: NamedTuple):
        self.path_images, self.path_labels, self.path_tfrecord, self.path_idxs = \
            args.path_images, args.path_labels, args.path_tfrecord, args.path_idxs
        assert(self.path_idxs.name[-3:].lower() == 'csv')
    
    def translate(self):
        writer = tf.io.TFRecordWriter(str(self.path_tfrecord))
        with open(str(self.path_idxs)) as csvfile:
            reader = csv.reader(csvfile)
            for image_name, label_name in tqdm(list(reader)):
                example = create_example(self.path_images.joinpath(image_name), self.path_labels.joinpath(label_name))
                writer.write(example.SerializeToString())
        writer.close()

import argparse
def parse_args():
    cvt2Path = lambda x: Path(x)
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-dataset", type=cvt2Path, default=Path("/home/yy/Coding/datasets/VOC/"),
        help="the path of the VOC dataset")
    parser.add_argument("--subset", type=str, default='train',
        help="the subset of the dataset to translate (train/val)")
    args = parser.parse_args()
    args.path_images = args.path_dataset.joinpath("images")
    args.path_labels = args.path_dataset.joinpath("labels")
    args.path_idxs = args.path_dataset.joinpath(f"{args.subset}.csv")
    assert(args.path_images.exists())
    assert(args.path_labels.exists())
    assert(args.path_idxs.exists())
    args.path_tfrecord = args.path_dataset.joinpath("tfrecord")
    args.path_tfrecord.mkdir(exist_ok=True)
    args.path_tfrecord = args.path_tfrecord.joinpath(f"VOC-{args.subset}.tfrecord")
    return args

if __name__ == '__main__':
    args = parse_args()
    translater = TFrecordTranslater(args)
    translater.translate()

