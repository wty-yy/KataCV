from katacv.utils.related_pkgs.utility import *
from katacv.yolov5.parser import YOLOv5Args, get_args_and_writer
from torch.utils.data import Dataset, DataLoader
from katacv.utils.coco.constant import MAX_NUM_BBOXES_TRAIN, MAX_NUM_BBOXES_VAL
import cv2
import numpy as np
from PIL import Image
import warnings
import random

from katacv.utils.yolo.utils import (
  xywh2xyxy, xywh2cxcywh, xyxy2cxcywh,
  transform_affine, transform_hsv, transform_pad, show_box
)

class YOLODataset(Dataset):
  def __init__(self, image_size: int, subset: str, path_dataset: Path):
    self.img_size = image_size
    self.path_dataset = path_dataset
    self.augment = False if subset == 'val' else True
    self.max_num_bboxes = (
      MAX_NUM_BBOXES_TRAIN if subset == 'train' else MAX_NUM_BBOXES_VAL
    )
    path_annotation = self.path_dataset.joinpath(f"{subset}_annotation.txt")
    paths = np.genfromtxt(str(path_annotation), dtype=np.str_)
    self.paths_img, self.paths_box = paths[:, 0], paths[:, 1]
  
  def __len__(self):
    return len(self.paths_img)
  
  @staticmethod
  def _check_bbox_need_placeholder(bboxes):
    if len(bboxes) == 0:
      bboxes = np.array([[0,0,1,1,-1]], dtype=np.float32)  # placeholder
    return bboxes
  
  def load_file(self, idx):
    path_img, path_box = self.paths_img[idx], self.paths_box[idx]
    img = np.array(Image.open(str(self.path_dataset.joinpath(path_img))).convert('RGB')).astype('uint8')
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      box = np.loadtxt(self.path_dataset.joinpath(path_box))
    if len(box):
      box = np.roll(box.reshape(-1, 5), -1, axis=1)
    else:
      box = box.reshape(0, 5)

    h0, w0 = img.shape[:2]
    r = self.img_size / max(h0, w0)
    if r != 1:  # resize the max aspect to image_size
      interp = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA  # enlarge or shrink
      img = cv2.resize(img, (math.ceil(w0 * r), math.ceil(h0 * r)), interpolation=interp)
      if len(box):
        box[:, [0,2]] *= img.shape[1] / w0
        box[:, [1,3]] *= img.shape[0] / h0
    return img, box, img.shape[:2]

  def mosaic4(self, idx):
    s = self.img_size
    idxs = [idx] + random.choices(range(len(self.paths_img)), k=3)
    random.shuffle(idxs)
    border = s // 2
    cx, cy = np.random.uniform(border, s+border, 2).astype(np.int32)
    img4, box4 = np.full((2*s, 2*s, 3), 114, dtype=np.uint8), []  # 114 is the img RGB mean averaged in ImageNet
    for i, idx in enumerate(idxs):
      img, box, (h, w) = self.load_file(idx)  # box: COCO fmt
      box = xywh2xyxy(box)
      if i == 0:
        bx2, by2, bx1, by1 = cx, cy, max(0, cx-w), max(0, cy-h)
        sx2, sy2, sx1, sy1 = w, h, w - (bx2-bx1) , h - (by2 - by1)
      elif i == 1:
        bx1, by2, by1, bx2 = cx, cy, max(0, cy-h), min(2*s, cx+w)
        sx1, sy2, sy1, sx2 = 0, h, h - (by2-by1), bx2 - bx1
      elif i == 2:
        bx2, by1, bx1, by2 = cx, cy, max(0, cx-w), min(2*s, cy+h)
        sx2, sy1, sx1, sy2 = w, 0, w - (bx2-bx1), by2 - by1
      else:
        bx1, by1, bx2, by2 = cx, cy, min(2*s, cx+w), min(2*s, cy+h)
        sx1, sy1, sx2, sy2 = 0, 0, bx2 - bx1, by2 - by1
      img4[by1:by2, bx1:bx2] = img[sy1:sy2, sx1:sx2]
      dx, dy = bx1 - sx1, by1 - sy1
      box4.append(box + np.array([[dx, dy, dx, dy, 0]]))
    box4 = np.concatenate(box4, axis=0)
    box4[:,:4] = np.clip(box4[:,:4], 0, 2*s)
    img4, box4 = transform_affine(img4, box4, border=border)
    box4 = xyxy2cxcywh(box4)
    return img4, box4

  def __getitem__(self, idx):
    if self.augment:
      img, box = self.mosaic4(idx)  # yolo format (R)
      img = transform_hsv(img)
      if random.random() < 0.5:  # Flip left-right
        img = np.fliplr(img)
        if len(box):
          box[:, 0] = img.shape[1] - box[:, 0]
    else:
      img, box, _ = self.load_file(idx)
      img, (dh, dw) = transform_pad(img, (self.img_size, self.img_size))
      box[:, 0] += dw
      box[:, 1] += dh
      box = xywh2cxcywh(box)

    pbox = np.zeros((self.max_num_bboxes, 5))
    if len(box):
      pbox[:len(box)] = box
    return img.copy(), pbox.copy(), len(box)

class DatasetBuilder:
  args: YOLOv5Args

  def __init__(self, args: YOLOv5Args):
    self.args = args
  
  def get_dataset(self, subset: str = 'val'):
    dataset = YOLODataset(image_size=self.args.image_shape[0], subset=subset, path_dataset=self.args.path_dataset)
    ds = DataLoader(
      dataset, batch_size=self.args.batch_size,
      shuffle=subset == 'train',
      num_workers=self.args.num_data_workers,
      drop_last=True,
    )
    return ds

if __name__ == '__main__':
  args = get_args_and_writer(no_writer=True)
  ds_builder = DatasetBuilder(args)
  args.batch_size = 1
  # ds = ds_builder.get_dataset(subset='train')
  ds = ds_builder.get_dataset(subset='val')
  print("Dataset size:", len(ds))
  iterator = iter(ds)
  # image, bboxes, num_bboxes = next(iterator)
  # image, bboxes, num_bboxes = image.numpy(), bboxes.numpy(), num_bboxes.numpy()
  # print(image.shape, bboxes.shape, num_bboxes.shape)
  # for image, bboxes, num_bboxes in tqdm(ds):
  #   image, bboxes, num_bboxes = image.numpy(), bboxes.numpy(), num_bboxes.numpy()
  for i in range(8):
    image, bboxes, num_bboxes = next(iterator)
    image, bboxes, num_bboxes = image.numpy(), bboxes.numpy(), num_bboxes.numpy()
    print(image.shape, bboxes.shape, num_bboxes.shape)
    show_box(image[0], bboxes[0][np.arange(num_bboxes[0])])
  