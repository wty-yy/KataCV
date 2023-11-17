# -*- coding: utf-8 -*-
'''
@File    : preprocess_raw_coco_dataset.py
@Time    : 2023/11/14 22:15:26
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : 
Preprocess the raw COCO dataset.
Train: 118287
Val: 5000
Total: 123288 

Useage:
  Download three files and unzip them in same folder:
  - `train2017.zip`: http://images.cocodataset.org/zips/train2017.zip
  - `val2017.zip`: http://images.cocodataset.org/zips/val2017.zip
  - `annotations_trainval2017.zip`: http://images.cocodataset.org/annotations/annotations_trainval2017.zip

  Modify `path_dataset` to your COCO dataset folder.

Result:
  The script will create one folder and two txt files in `path_dataset`:
  - (folder) `bboxes`: 123288 txt files contain the bboxes parameters with `COCO` format.
  - (txt) `train_annotation.txt`: 118286 lines, each line gives the \
    `image_path` and `bbox_path`.
  - (txt) `val_annotation.txt`: 5000 lines, each line gives the \
    `image_path` and `bbox_path`.
  
  Also create `label2name.py` file in current script folder.

'''
import json
from pathlib import Path
from tqdm import tqdm
import math

path_dataset = Path('/home/wty/Coding/datasets/coco')
path_annotation = path_dataset.joinpath("annotations")
path_bboxes = path_dataset.joinpath("bboxes")
path_bboxes.mkdir(exist_ok=True)

def make_bboxes_files(subset="train"):
  path_instance = path_annotation.joinpath(f"instances_{subset}2017.json")
  print("Loading JSON file...")
  with open(path_instance, 'r') as file:
    instance = json.load(file)
  print("Loading JSON finished")
  categories = instance['categories']
  category2id = {categories[i]['id']: i for i in range(len(categories))}
  id2name = {i: categories[i]['name'] for i in range(len(categories))}
  # imageId2imageShape = {x['id']: (x['height'], x['width']) for x in instance['images']}
  imageId2bboxes = {x['id']: [] for x in instance['images']}
  for annotation in instance['annotations']:
    x, y, w, h = annotation['bbox']
    # shape = imageId2imageShape[annotation['image_id']]
    # x = round(x + w / 2, 2) / shape[1]
    # y = round(y + h / 2, 2) / shape[0]
    # w = math.floor(w / shape[1] * 100) / 100
    # h = math.floor(h / shape[0] * 100) / 100
    imageId2bboxes[annotation['image_id']].append((
      category2id[annotation['category_id']],
      x, y, w, h
    ))
  bar = tqdm(sorted(imageId2bboxes.items(), key=lambda a: a[0]), total=len(imageId2bboxes))
  file_annotation = open(path_dataset.joinpath(f"{subset}_annotation.txt"), 'w')
  datasize = len(imageId2bboxes)
  max_num_bboxes = 0
  for image_id, bboxes in bar:
    with open(path_bboxes.joinpath(f"{image_id:012}.txt"), 'w') as file:
      for bbox in bboxes:
        max_num_bboxes = max(max_num_bboxes, len(bboxes))
        file.write(" ".join([str(x) for x in bbox]))
        file.write('\n')
    file_annotation.write(f"./{subset}2017/{image_id:012}.jpg ./bboxes/{image_id:012}.txt")
    file_annotation.write('\n')
  file_annotation.close()
  print(f"Data size of {subset}: {datasize}")
  print(f"Maximum number of {subset} bounding boxes: {max_num_bboxes}")
  with open("./constant.py", 'w') as file:
    file.write("label2name = {\n")
    for id, name in id2name.items():
      file.write(f"  {id}: '{name}',\n")
    file.write("}\n")
  return datasize, max_num_bboxes

if __name__ == '__main__':
  datasize_val, max_num_bboxes_val = make_bboxes_files(subset='val')
  datasize_train, max_num_bboxes_train = make_bboxes_files(subset='train')
  with open("./constant.py", 'a') as file:
    file.write(f"MAX_NUM_BBOXES_TRAIN = {max_num_bboxes_train}\n")
    file.write(f"DATASIZE_TRAIN = {datasize_train}\n")
    file.write(f"MAX_NUM_BBOXES_VAL = {max_num_bboxes_val}\n")
    file.write(f"DATASIZE_VAL = {datasize_val}\n")