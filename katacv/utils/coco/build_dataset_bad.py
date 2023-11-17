from katacv.utils.related_pkgs.utility import *
from katacv.yolov4.parser import YOLOv4Args, get_args_and_writer
import tensorflow as tf

class DatasetBuilder:
  args: YOLOv4Args
  
  def __init__(self, args: YOLOv4Args):
    self.args = args
  
  def convert_annotation_file_to_dataset(self, subset) -> tf.data.Dataset:
    path_images, path_bboxes = [], []
    with open(self.args.path_dataset.joinpath(f"{subset}_annotation.txt"), 'r') as file:
      for line in file.readlines():
        if len(line) == 0: continue
        path1, path2 = line.split(' ')
        path_images.append(str(self.args.path_dataset.joinpath(path1)))
        path_bboxes.append(str(self.args.path_dataset.joinpath(path2)))
    return tf.data.Dataset.from_tensor_slices((path_images, path_bboxes))
  
  def decoder(self, path_image, path_params):
    sample = {'image': None, 'bboxes': [], 'classes': []}

    def process_params(param):
      param = tf.strings.strip(tf.strings.split(param, ' '))
      sample['classes'].append(tf.strings.to_number(param[0], out_type=tf.int32))
      tmp = []
      for i in range(1, 5):
        tmp.append(tf.strings.to_number(param[i], out_type=tf.float32))
      bbox = tf.expand_dims(tf.stack(tmp, axis=0), 0)
      sample['bboxes'].append(bbox)

    sample['image'] = tf.io.decode_jpeg(tf.io.read_file(path_image), channels=3)
    params = tf.io.read_file(path_params)
    sample['bboxes'] = params
    return sample
    tf.py_function(func=lambda param: process_params(param), inp=[params], Tout=[])
    
    # for param in tf.strings.split(params, '\n'):
    #   param = tf.strings.strip(tf.strings.split(param, ' '))
    #   sample['classes'].append(tf.strings.to_number(param[0], out_type=tf.float32))
    #   tmp = []
    #   for i in range(1, 5):
    #     tmp.append(tf.strings.to_number(param[i], out_type=tf.float32))
    #   bbox = tf.expand_dims(tf.stack(bbox, axis=0), 0)
    #   sample['bboxes'].append(bbox)
    if len(sample['bboxes']) == 0:
      sample['bboxes'].append(tf.zeros((1, 4), tf.float32))
      sample['classes'].append(tf.constant(-1, tf.int32))
    sample['bboxes'] = tf.concat(sample['bboxes'], axis=0)
    sample['classes'] = tf.stack(sample['classes'], axis=0)
    return sample
  
  def get_dataset(self, subset='train', shuffle=True, use_aug=True):
    ds_paths = self.convert_annotation_file_to_dataset(subset)
    ds = ds_paths.map(self.decoder, num_parallel_calls=tf.data.AUTOTUNE, deterministic=False)
    return ds

if __name__ == '__main__':
  args = get_args_and_writer(no_writer=True)
  ds_builder = DatasetBuilder(args)
  ds = ds_builder.get_dataset(subset='val')
  for sample in ds:
    print(sample['image'].shape, sample['bboxes'].shape, sample['classes'].shape)
    break
