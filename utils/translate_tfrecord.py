# -*- coding: utf-8 -*-
'''
@File    : translate_tfrecord.py
@Time    : 2023/09/08 21:32:50
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : 
该文件用于将dataset转化为TFRecord二进制形式，用于快速读入，需要先使用utils/make_label_json.py生成类别名称的对应编号json文件
- 转化Imagenet2012数据集用时13:12:49，最终文件大小147GB
'''
import tensorflow as tf, json, argparse
from pathlib import Path
from tqdm import tqdm
import numpy as np

path_origin_dataset = Path("/media/yy/Data/dataset/imagenet")
path_logs = Path.cwd().joinpath("logs")
def parse_args():
    cvt2path = lambda x: Path(x)
    parser = argparse.ArgumentParser()
    parser.add_argument("--path-origin-dataset", type=cvt2path, default=path_origin_dataset,
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
    args.path_subdataset = args.path_origin_dataset.joinpath(args.subfolder_name)
    args.path_tfrecord = args.path_origin_dataset.joinpath(f"tfrecord/imagenet2012-{args.subfolder_name}-origin.tfrecord")
    assert(args.path_logs.exists())
    assert(args.path_origin_dataset.exists())
    assert(args.path_subdataset.exists())
    return args

def translate_tfrecord(args):
    image_paths = []
    for subclass in sorted(args.path_subdataset.iterdir()):
        if subclass.is_dir():
            for image_path in subclass.iterdir():
                if image_path.name[-4:].lower() == args.image_type:
                    image_paths.append((str(image_path), nameid2label[subclass.name]))
    print(f"find images for {args.subfolder_name}:", len(image_paths))

    def serialize_example(path, label):
        bytes_image = tf.io.read_file(path)
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'image': tf.train.Feature(bytes_list=tf.train.BytesList(value=[bytes_image.numpy()])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label]))
            }
        ))
        return example.SerializeToString()

    image_paths = np.array(image_paths)
    np.random.shuffle(image_paths)

    writer = tf.io.TFRecordWriter(str(args.path_tfrecord))
    for path, label in tqdm(image_paths):
        writer.write(serialize_example(path, int(label)))
    writer.close()

if __name__ == '__main__':
    args = parse_args()
    np.random.seed(args.seed)
    with open(args.path_logs.joinpath("nameid2label.json"), 'r') as file:
        nameid2label = json.load(file)
    translate_tfrecord(args)