# -*- coding: utf-8 -*-
'''
@File    : build_dataset.py
@Time    : 2023/11/18 00:53:17
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : 
BUG: The bboxes file:
`./bboxes/000000200365.txt` line 3,
`./bboxes/000000550395.txt` line 15
have 0 height, we should remove it.
Had been solved by data YOLODataset._load_data(index)
'''
from katacv.utils.related_pkgs.utility import *
from katacv.yolov4.parser import YOLOv4Args, get_args_and_writer
from torch.utils.data import Dataset, DataLoader
from katacv.utils.coco.constant import MAX_NUM_BBOXES_TRAIN, MAX_NUM_BBOXES_VAL
import albumentations as A
import cv2
import numpy as np
from PIL import Image
import warnings

class YOLODataset(Dataset):
  args: YOLOv4Args
  subset: str
  shuffle: bool
  transform: Callable
  path_images: np.ndarray
  path_bboxes: np.ndarray
  use_mosaic4: bool
  max_num_bboxes: int

  def __init__(
      self, args: YOLOv4Args, subset: str, shuffle: bool,
      transform: Callable, use_mosaic4: bool
    ):
    self.args, self.subset, self.shuffle, self.transform, self.use_mosaic4 = (
      args, subset, shuffle, transform, use_mosaic4
    )
    self.max_num_bboxes = (
      MAX_NUM_BBOXES_TRAIN if subset == 'train' else MAX_NUM_BBOXES_VAL
    )
    path_annotation = self.args.path_dataset.joinpath(f"{subset}_annotation.txt")
    paths = np.genfromtxt(str(path_annotation), dtype=np.str_)
    self.path_images, self.path_bboxes = paths[:, 0], paths[:, 1]
  
  def __len__(self):
    return len(self.path_images)
  
  @staticmethod
  def _check_bbox_need_placeholder(bboxes):
    if len(bboxes) == 0:
      bboxes = np.array([[0,0,1,1,-1]], dtype=np.float32)  # placeholder
    return bboxes
  
  def _load_data(self, index):
    path_image = self.args.path_dataset.joinpath(self.path_images[index])
    image = np.array(Image.open(path_image).convert("RGB"))
    path_bboxes = self.args.path_dataset.joinpath(self.path_bboxes[index])
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")
      bboxes = np.loadtxt(path_bboxes)
    if len(bboxes):
      # bboxes parameters: (x, y, w, h, class_id)
      bboxes = np.roll(bboxes.reshape(-1, 5), -1, axis=1)
    bboxes = self._check_bbox_need_placeholder(bboxes)
    bboxes = bboxes[(bboxes[:,2]>0)&(bboxes[:,3]>0)]  # avoid 0 wide for two data in COCO
    return image, bboxes
  
  def _mosaic_transform4(self, x0, x1, x2, x3, y0, y1, y2, y3, center_rate=0.2):
    def crop(x, y, h, w):
      transformed = A.Compose(
        [
          A.RandomCrop(h, w),
        ],
        bbox_params=A.BboxParams(format='coco', min_visibility=0.4)
      )(image=x, bboxes=y)
      x, y = transformed['image'], np.array(transformed['bboxes'])
      y = self._check_bbox_need_placeholder(y)
      return x, y
    H, W = self.args.image_shape[:2]
    # x = np.zeros(self.args.image_shape, dtype=np.float32)
    center_h = int(np.random.uniform(-H*center_rate, H*center_rate) + H / 2.0)
    center_w = int(np.random.uniform(-W*center_rate, W*center_rate) + W / 2.0)
    # center = np.stack([center_h, center_w])
    x0, y0 = crop(
      x0, y0,
      center_h, center_w
    )
    x1, y1 = crop(
      x1, y1,
      center_h, W-center_w
    )
    x2, y2 = crop(
      x2, y2,
      H-center_h, center_w
    )
    x3, y3 = crop(
      x3, y3,
      H-center_h, W-center_w
    )
    y1[:,0] += center_w; y3[:,0] += center_w
    y2[:,1] += center_h; y3[:,1] += center_h
    # x[:center_h, :center_w, :] = x0
    # x[:center_h, center_w:, :] = x1
    # x[center_h:, :center_w, :] = x2
    # x[center_h:, center_w:, :] = x3
    print(x0.shape, x1.shape, x2.shape, x3.shape)
    x = np.concatenate([
      np.concatenate([x0,x1], axis=1), np.concatenate([x2,x3], axis=1)
    ], axis=0)
    y = np.concatenate([y0,y1,y2,y3], axis=0)
    y = y[y[:,4] != -1]
    y = self._check_bbox_need_placeholder(y)
    return x, y
  
  def _transform(self, image, bboxes):
      try:
        transformed = self.transform(image=image, bboxes=bboxes)
      except Exception as e:
        print("Error Id:", self.path_bboxes[index])
        raise e
      image, bboxes = transformed['image'], np.array(transformed['bboxes'])
      return image, bboxes
  
  def __getitem__(self, index):
    image, bboxes = self._load_data(index)
    # Maybe remove all the bboxes after transform
    if self.transform:
      image, bboxes = self._transform(image, bboxes)
    bboxes = self._check_bbox_need_placeholder(bboxes)

    if self.use_mosaic4:
      xs, ys = [image], [bboxes]
      for i in np.random.randint(0, len(self.path_images), size=3):
        x, y = self._transform(*self._load_data(i))
        xs.append(x); ys.append(y)
      image, bboxes = self._mosaic_transform4(*xs, *ys)
      # print(image.shape, bboxes.shape)
      # print(bboxes)

    # Convert output to yolo format, xy to the center! 2023/11/20
    bboxes[:,:2] += bboxes[:,2:4] / 2
    num_bboxes = np.sum(bboxes[:,4] != -1)
    bboxes = np.concatenate([
      bboxes,
      np.repeat(
        [(0, 0, 1, 1, -1)],
        repeats=self.max_num_bboxes-bboxes.shape[0], axis=0
    )], axis=0)
    return image, bboxes, num_bboxes

class DatasetBuilder:
  args: YOLOv4Args

  def __init__(self, args: YOLOv4Args):
    self.args = args
  
  def get_transform(self, subset):
    scale = 1.4
    train_transform = A.Compose(
      [
        A.RandomResizedCrop(
          height=int(self.args.image_shape[1]),
          width=int(self.args.image_shape[0]),
          scale=(0.8, 1),
        ),
        # A.LongestMaxSize(max_size=int(max(self.args.image_shape[:2])*scale)),
        # A.PadIfNeeded(
        #   min_height=int(self.args.image_shape[0]*scale),
        #   min_width=int(self.args.image_shape[1]*scale),
        #   border_mode=cv2.BORDER_CONSTANT,
        # ),
        A.ColorJitter(brightness=0.4, contrast=0.0, saturation=0.7, hue=0.015, p=0.4),
        # A.OneOf(
        #   [
        #     A.ShiftScaleRotate(
        #       rotate_limit=10, p=0.5, border_mode=cv2.BORDER_CONSTANT
        #     ),
        #     A.Affine(shear=10, p=0.5, mode=cv2.BORDER_CONSTANT),
        #   ], p=0.4
        # ),
        A.HorizontalFlip(p=0.5),
        # A.Blur(p=0.1),
        # A.Posterize(p=0.1),
        A.ToGray(p=0.05),
        # A.ChannelShuffle(p=0.01),
        A.RandomCrop(*self.args.image_shape[:2]),
        # A.CenterCrop(*self.args.image_shape[:2]),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
      ],
      bbox_params=A.BboxParams(format='coco', min_visibility=0.4)
    )
    val_transform = A.Compose(
      [
        A.LongestMaxSize(max_size=self.args.image_shape[:2]),
        A.PadIfNeeded(
          min_height=self.args.image_shape[0],
          min_width=self.args.image_shape[1],
          border_mode=cv2.BORDER_CONSTANT,
        ),
        A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
      ],
      bbox_params=A.BboxParams(format='coco', min_visibility=0.4)
    )
    return train_transform if subset == 'train' else val_transform
    # return val_transform if subset == 'val' else train_transform
  
  def get_dataset(self, subset: str = 'val', shuffle: bool = True):
    dataset = YOLODataset(
      self.args, subset, shuffle, self.get_transform(subset),
      use_mosaic4=False if subset == 'val' or not self.args.use_mosaic4 else True
    )
    ds = DataLoader(
      dataset, batch_size=self.args.batch_size,
      shuffle=subset == 'train',
      num_workers=self.args.num_data_workers,
      drop_last=True,
    )
    return ds

def show_bbox(image, bboxes, draw_center_point=False):
  from katacv.utils.detection import plot_box_PIL, build_label2colors
  from katacv.utils.coco.constant import label2name
  if type(image) != Image.Image:
    image = Image.fromarray((image*255).astype('uint8'))
  if len(bboxes):
    label2color = build_label2colors(bboxes[:,4])
  for bbox in bboxes:
    label = int(bbox[4])
    image = plot_box_PIL(image, bbox[:4], text=label2name[label], box_color=label2color[label], format='yolo', draw_center_point=draw_center_point)
    print(label, label2name[label], label2color[label])
  image.show()

if __name__ == '__main__':
  args = get_args_and_writer(no_writer=True)
  ds_builder = DatasetBuilder(args)
  args.batch_size = 1
  ds = ds_builder.get_dataset(subset='train')
  print("Dataset size:", len(ds))
  iterator = iter(ds)
  # image, bboxes, num_bboxes = next(iterator)
  # image, bboxes, num_bboxes = image.numpy(), bboxes.numpy(), num_bboxes.numpy()
  # print(image.shape, bboxes.shape, num_bboxes.shape)
  # for image, bboxes, num_bboxes in tqdm(ds):
  #   image, bboxes, num_bboxes = image.numpy(), bboxes.numpy(), num_bboxes.numpy()
  for i in range(10):
    image, bboxes, num_bboxes = next(iterator)
    image, bboxes, num_bboxes = image.numpy(), bboxes.numpy(), num_bboxes.numpy()
    print(image.shape, bboxes.shape, num_bboxes.shape)
    show_bbox(image[0], bboxes[0][np.arange(num_bboxes[0])])
  