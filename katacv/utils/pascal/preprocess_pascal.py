# -*- coding: utf-8 -*-
'''
@File    : preprocess_pascal.py
@Time    : 2023/11/20 15:45:05
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : 
Download PASCAL VOC 2007/2012 dataset from:
- https://www.kaggle.com/datasets/734b7bcb7ef13a045cbdd007a3c19874c2586ed0b02b4afc86126e89d00af8d2

Result:
train datasize: 16550
train max bboxes num: 39
val datasize: 4951
val max bboxes num: 37
'''
from katacv.utils.related_pkgs.utility import *
import pandas as pd
from PIL import Image
import numpy as np

path_dataset = Path(r"/home/wty/Coding/datasets/PASCAL")

def check_dataset(subset='train'):
  df = pd.read_csv(path_dataset.joinpath(subset+".csv"))
  print(f"{subset} datasize:", df.shape[0])
  max_bboxes_num = 0
  for i in tqdm(range(df.shape[0])):
    path_image, path_label = df.iloc[i,0], df.iloc[i,1]
    try:
      image = Image.open(str(path_dataset.joinpath("images/"+path_image)))
    except Exception as e:
      print(f"Error: Image {str(path_image)} can't be load.")
    labels = np.loadtxt(str(path_dataset.joinpath("labels/"+path_label)))
    max_bboxes_num = max(max_bboxes_num, labels.shape[0])
  print(f"{subset} max bboxes num: {max_bboxes_num}")

check_dataset('train')
check_dataset('val')
