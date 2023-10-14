# -*- coding: utf-8 -*-
'''
@File    : translate_tfrecord.py
@Time    : 2023/10/14 09:45:02
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : 构建OCR所需的数据集
MJSynth数据集: https://www.robots.ox.ac.uk/~vgg/data/text/#sec-synth，
其中`annotation_train.txt`, `annotation_val.txt`给出了对应的图片地址和其对应的文本，格式如下：

directory1 text_label1
directory2 text_label2
directory3 text_label3
...

将将其转化为tfrecord准备后续读取。
'''
from pathlib import Path
from typing import NamedTuple
import tensorflow as tf

def create_example(path_image, label):
    image_bytes = tf.io.read_file(str(path_image))
    image = tf.io.decode_jpeg(image_bytes, channels=1)
    image_bytes = tf.io.encode_jpeg(image)
    label_bytes = label.encode('utf-8')
    
    feature_dict = {
        'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_bytes.numpy()])),
        'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[label_bytes])),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example

from tqdm import tqdm

class TFrecordTranslater:
    path_tfrecord: Path  # save the tfrecord
    path_dataset: Path  # dataset
    path_idxs: Path  # txt

    def __init__(self, args: NamedTuple):
        self.path_tfrecord, self.path_dataset, self.path_idxs = args.path_tfrecord, args.path_dataset, args.path_idxs
        assert(self.path_idxs.name[-3:].lower() == 'txt')
    
    def translate(self):
        writer = tf.io.TFRecordWriter(str(self.path_tfrecord))
        with open(str(self.path_idxs)) as file:
            for line in tqdm(list(file)):
                path, _ = line.strip().split(' ')
                image_path = self.path_dataset.joinpath(path)
                label = image_path.name.split('_')[1]
                try:
                    example = create_example(image_path, label)
                except:
                    print("Wrong image path:", image_path.absolute())
                    continue
                writer.write(example.SerializeToString())
        writer.close()

import argparse
def parse_args():
    cvt2Path = lambda x: Path(x)
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-dataset", type=cvt2Path, default=Path("/home/wty/Coding/datasets/mjsynth/"),
        help="the path of the mjsynth dataset")
    parser.add_argument("--subset", type=str, default='val',
        help="the subset of the dataset to translate (train/val)")
    args = parser.parse_args()
    args.path_idxs = args.path_dataset.joinpath(f"annotation_{args.subset}.txt")
    assert(args.path_idxs.exists())
    args.path_tfrecord = args.path_dataset.joinpath("tfrecord")
    args.path_tfrecord.mkdir(exist_ok=True)
    args.path_tfrecord = args.path_tfrecord.joinpath(f"mjsynth-{args.subset}.tfrecord")
    return args

if __name__ == '__main__':
    args = parse_args()
    translater = TFrecordTranslater(args)
    translater.translate()

