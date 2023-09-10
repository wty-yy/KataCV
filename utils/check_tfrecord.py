# -*- coding: utf-8 -*-
'''
@File    : translate_tfrecord_check.py
@Time    : 2023/09/10 21:29:04
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : 
对translate_tf_record.py保存好的tfrecord进行文件检查，并统计完好图片数目
'''

if __name__ == '__main__':
    pass

import tensorflow as tf, json, argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np

path_origin_tfrecord = Path("/media/yy/Data/dataset/imagenet")
path_logs = Path.cwd().joinpath("logs")
def parse_args():
    cvt2path = lambda x: Path(x)
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-origin-tfrecord", type=cvt2path, default=path_origin_tfrecord,
        help="the path of the original dataset (JPEG)")
    parser.add_argument("--path-logs", type=cvt2path, default=path_logs,
        help="the path for saving the json logs")
    parser.add_argument("--subfolder-name", type=str, default="train",
        help="the subfolder name to translate those images to TFRecord")
    parser.add_argument("--image-type", type=str, default="jpeg",
        help="the type of the images (jpeg, jpg, png, ...)")
    parser.add_argument("--seed", type=int, default=1,
        help="the seed for shuffle the images' order")
    args = parser.parse_args()
    args.path_sub_origin_tfrecord = args.path_origin_tfrecord.joinpath(f"imagenet2012-{args.subfolder_name}-origin.tfrecord")
    args.path_tfrecord = args.path_origin_tfrecord.joinpath(f"imagenet2012-{args.subfolder_name}.tfrecord")
    assert(args.path_logs.exists())
    assert(args.path_origin_tfrecord.exists())
    assert(args.path_sub_origin_tfrecord.exists())
    return args

def decode_jpeg_tfrecord(example):
    feature = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64),
    }
    example = tf.io.parse_single_example(example, feature)
    image = tf.io.decode_jpeg(example['image'], channels=3)
    return image, example['label']

def check_tfrecord(args):
    ds_tfrecord = tf.data.TFRecordDataset(args.path_sub_origin_tfrecord)
    ds = ds_tfrecord.map(decode_jpeg_tfrecord).ignore_errors(log_warning=True)

    def serialize_example(image, label):
        bytes_image = tf.io.encode_jpeg(image)
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes_image.numpy()])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            }
        ))
        return example.SerializeToString()

    writer = tf.io.TFRecordWriter(str(args.path_tfrecord))
    count = 0
    for image, label in tqdm(ds):
        writer.write(serialize_example(image, int(label)))
        count += 1
    print(f"Available image in '{args.path_sub_origin_tfrecord}':", count)
    writer.close()

if __name__ == '__main__':
    args = parse_args()
    np.random.seed(args.seed)
    with open(args.path_logs.joinpath("nameid2label.json"), 'r') as file:
        nameid2label = json.load(file)
    check_tfrecord(args)