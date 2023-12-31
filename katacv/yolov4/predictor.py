# -*- coding: utf-8 -*-
'''
@File    : predictor.py
@Time    : 2023/12/03 13:52:59
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : 
2023/12/03: YOLOv4-0300:
P@50=0.5012 R@50=0.3885 AP@50=0.3507 AP@75=0.1545 mAP=0.1729: 100%|████| 156/156 [00:31<00:00,  4.95it/s]
'''
from katacv.utils.imagenet.train import TrainState
from katacv.utils.related_pkgs.utility import *
from katacv.utils.related_pkgs.jax_flax_optax_orbax import *
from katacv.yolov4.parser import YOLOv4Args
from katacv.yolov4.build_yolo_target import cell2pixel
from katacv.yolov4.metric import logits2cell, logits2prob_from_list, nms, show_bbox
from katacv.utils.detection import iou_multiply
from katacv.utils.detection.utils_ap import ap_per_class
import numpy as np

class Predictor:
  """
  Predict and metrics for the YOLO model.

  Args:
    args: YOLO model argmentations.
    state: YOLO model state.
    pbox: Predicted bounding boxes. List[box.shape=(M,6), elem=(x,y,w,h,conf,cls)]
    tcls: Class of target bounding boxes. List[cls.shape=(M',)]
    tp: Ture positive for the `pbox`. List[tp.shape=(M,len(iout))]
    iout: The threshold of iou for deciding whether is the ture positive. List[int]
  """
  args: YOLOv4Args
  state: TrainState
  pbox: List[np.ndarray]  # np.float32
  tcls: List[np.ndarray]  # np.int32
  tp: List[np.ndarray]  # np.bool_
  iout: jax.Array

  def __init__(self, args: YOLOv4Args, state: TrainState, iout=None):
    self.args, self.state = args, state
    self.iout = jnp.linspace(0.5, 0.95, 10) if iout is None else iout
    if type(self.iout) == float:
      self.iout = jnp.array([self.iout,])
    self.reset()
  
  def reset(self):
    self.pbox, self.tcls, self.tp = [], [], []
  
  def update(self, x, tbox=None, tnum=None, nms_iou=0.6, nms_conf=0.001):
    """
    Update the prediction variables.

    Args: (`tbox`, `tnum` and `B` can be `None`, \
        just predict the bounding boxes for `x`), \
        `B` is the number of the batch size.
      x: The input of the model. [shape=(B,H,W,C) or (H,W,C)]
      tbox: The target bounding boxes. [shape=(B,M,5), or (M,5)]
      tnum: The number of the target bounding boxes. [shape=(B,) or int]
      nms_iou: The threshold of the iou in NMS.
      nms_conf: The threshold of the confidence in NMS.
    """
    if x.ndim == 3: x = x[None,...]
    if tbox is None and tnum is None:
      pbox, pnum = jax.device_get(self.pred_and_nms(x, nms_iou, nms_conf))
    else:
      assert(tbox is not None and tnum is not None)
      if tbox.ndim == 2: tbox = tbox[None,...]
      if type(tnum) == int: tnum = jnp.array((tnum,))
      pbox, pnum, tp = jax.device_get(self.pred_and_nms_and_tp(
        x, nms_iou, nms_conf, tbox, tnum
      ))
    for i in range(x.shape[0]):
      self.pbox.append(pbox[i][:pnum[i]])
      self.tcls.append(tbox[i][:tnum[i],4].astype(np.int32))
      self.tp.append(tp[i][:pnum[i]])
  
  def ap_per_class(self):
    """
    Compute average percision (AP) by \
      the area of under recall and precision curve (AUC) for each class.

    Return:
      p: Precision for each class with confidence bigger than 0.1. [shape=(Nc,)]
      r: Recall for each class with confidence bigger than 0.1. [shape=(Nc,)]
      ap: Average precision for each class with different iou thresholds. [shape=(Nc,tp.shape[1])]
      f1: F1 coef for each class with confidence bigger than 0.1. [shape=(Nc,)]
      ucls: Class labels after being uniqued. [shape=(Nc,)]
    """
    pbox = np.concatenate(self.pbox, axis=0)
    return ap_per_class(
      tp=np.concatenate(self.tp, axis=0),
      conf=pbox[:,4],
      pcls=pbox[:,5],
      tcls=np.concatenate(self.tcls, axis=0)
    )
  
  def p_r_ap50_ap75_map(self):
    """
    Return:
      p50: Precision with 0.5 iou threshold and bigger than 0.1 confidence.
      r50: Recall with 0.5 iou threshold and bigger than 0.1 confidence.
      ap50: Average precision by AUC with 0.5 iou threshold.
      ap75: Average precision by AUC with 0.75 iou threshold.
      map: Mean average precision by AUC with mean of 10 \
        different iou threshold [0.5:0.05:0.95].
    """
    p, r, ap = self.ap_per_class()[:3]
    p50, r50, ap50, ap75, ap = p[:,0], r[:,0], ap[:,0], ap[:,5], ap.mean(1)
    p50, r50, ap50, ap75, map = p50.mean(), r50.mean(), ap50.mean(), ap75.mean(), ap.mean()
    return p50, r50, ap50, ap75, map

  @partial(jax.jit, static_argnums=0)
  def predict(self, x: jax.Array):
    logits = self.state.apply_fn(
      {'params': self.state.params, 'batch_stats': self.state.batch_stats},
      x, train=False
    )
    pred_cell = [logits2cell(logits[i]) for i in range(3)]
    pred_pixel = [
      jax.vmap(cell2pixel, in_axes=(0,None,None), out_axes=0)(
        pred_cell[i], 2**(i+3), self.args.anchors[i]
      ) for i in range(3)
    ]
    pred_pixel_prob = logits2prob_from_list(pred_pixel)
    return pred_pixel_prob

  @partial(jax.jit, static_argnums=[0,2,3])
  def pred_and_nms(
    self, x: jax.Array,
    iou_threshold: float, conf_threshold: float
  ):
    pred = self.predict(x)
    pbox, pnum = jax.vmap(
      nms, in_axes=[0, None, None], out_axes=0
    )(pred, iou_threshold, conf_threshold)
    return pbox, pnum
  
  @partial(jax.jit, static_argnums=0)
  def pred_and_nms_and_tp(
    self, x: jax.Array,
    iou_threshold: float, conf_threshold: float,
    tbox: jax.Array, tnum: jax.Array
  ):
    pbox, pnum = self.pred_and_nms(x, iou_threshold, conf_threshold)
    pbox, tp = jax.vmap(self.compute_tp, in_axes=[0,0,0,0,None], out_axes=0)(
      pbox, pnum, tbox, tnum, self.iout
    )
    return pbox, pnum, tp
  
  @staticmethod
  @jax.jit
  def compute_tp(pbox, pnum, tbox, tnum, iout):
    """
    Compute the true positive for each `pbox`. Time complex: O(NM)

    Args:
      pbox: The predicted bounding boxes. [shape=(N,6), elem=(x,y,w,h,conf,cls)]
      pnum: The number of available `pbox`. [int]
      tbox: The target bounding boxes. [shape=(M,5), elem=(x,y,w,h,cls)]
      tnum: The number of available `tbox`. [int]
      iout: The iou thresholds of deciding true positive. [shape=(1,) or (10,)]
      num_classes: The number of all classes. [int]
    
    Return:
      pbox: The input `pbox` rearrange by increasing confidence.
      tp: The ture positive for the `pbox` after rearrange
    """
    sort_i = jnp.argsort(-pbox[:,4])
    pbox = pbox[sort_i]  # Decrease by confidence
    tp = jnp.zeros((pbox.shape[0],iout.shape[0]), jnp.bool_)
    def solve(tp):  # If tnum > 0
      ious = iou_multiply(pbox[:,:4], tbox[:,:4])  # shape=(N,M)
      bel = jnp.zeros((pbox.shape[0],))  # tbox index that pbox belong to
      # Get tp and belong
      def loop_i_fn(i, value):  # i=0,...,pnum-1
        tp, bel = value
        iou = ious[i]
        j = jnp.argmax((tbox[:,4]==pbox[i,5]) * iou)  # belong to tbox[j]
        # Round to 0.01, https://github.com/rafaelpadilla/Object-Detection-Metrics
        tp = tp.at[i].set((iou[j].round(2) >= iout - 1e-5) & (j < tnum))
        bel = bel.at[i].set(j)
        return tp, bel
      tp, bel = jax.lax.fori_loop(0, pnum, loop_i_fn, (tp, bel))
      # Remove duplicate belong
      bel = jnp.where(tp.sum(1) > 0, bel, -1)
      mask = jnp.zeros_like(bel, jnp.bool_)
      def loop_j_fn(j, mask):  # j=0,...,tnum-1
        tmp = bel == j
        tmp = tmp.at[jnp.argmax(tmp)].set(False)
        mask = mask | tmp
        return mask
      mask = jax.lax.fori_loop(0, tnum, loop_j_fn, mask)
      tp = tp * (~mask[:,None])
      return tp
    tp = jax.lax.cond(tnum > 0, solve, lambda x: x, tp)
    return pbox, tp

def load_pred_and_target_file(path_pred: Path, path_tg: Path, format='coco'):
  cls2idx = {'_num': 0}
  def load_file(path: Path):
    box = []
    with open(path, 'r') as file:
      for line in file.readlines():
        if len(line) == 0: continue
        line = line.split()
        if len(line) == 6:
          cls, conf, x, y, w, h = [x if i == 0 else float(x) for i, x in enumerate(line)]
        else:
          cls, x, y, w, h = [x if i == 0 else float(x) for i, x in enumerate(line)]
        if format == 'coco':
          x += w / 2
          y += h / 2
        if cls not in cls2idx:
          cls2idx[cls] = cls2idx['_num']; cls2idx['_num'] += 1
        cls = cls2idx[cls]
        if len(line) == 6:
          box.append(jnp.stack([x, y, w, h, conf, cls]))
        if len(line) == 5:
          box.append(jnp.stack([x, y, w, h, cls]))
    return jnp.stack(box)
  def load_dir(path_dir: Path):
    box, num = [], []
    for path in sorted(path_dir.iterdir()):
      if path.is_file():
        box.append(load_file(path))
        num.append(box[-1].shape[0])
    for i in range(len(box)):
      box[i] = jnp.pad(box[i], ((0, max(num) - box[i].shape[0]), (0, 0)))
    return jnp.stack(box), jnp.stack(num)
  pbox, pnum = load_dir(path_pred)
  tbox, tnum = load_dir(path_tg)
  return pbox, pnum, tbox, tnum, cls2idx

def metric_from_file(path_tg, path_pred):
  # Data from: https://github.com/rafaelpadilla/Object-Detection-Metrics
  pbox, pnum, tbox, tnum, cls2idx = load_pred_and_target_file(path_pred, path_tg)
  iout = jnp.linspace(0.3, 0.75, 10)  # just test
  # i = 2
  # pbox, tp = Predictor.compute_tp(pbox[i], pnum[i], tbox[i], tnum[i], iout)
  # print(pbox[:pnum[i]])
  # print(tp[:pnum[i]])
  pbox, tp = jax.vmap(
    Predictor.compute_tp, in_axes=[0,0,0,0,None], out_axes=0
  )(pbox, pnum, tbox, tnum, iout)
  p = Predictor(args=None, state=None)
  for i in range(pbox.shape[0]):
    p.pbox.append(pbox[i][:pnum[i]])
    p.tcls.append(tbox[i][:tnum[i],4].astype(np.int32))
    p.tp.append(tp[i][:pnum[i]])
    # print("idx: ", i)
    # print(pbox[i][:pnum[i]])
    # print(tp[i][:pnum[i]])
  print(p.ap_per_class())
  print(p.p_r_ap50_ap75_map())

def metric_from_model():
  from katacv.yolov4.parser import get_args_and_writer
  args = get_args_and_writer(no_writer=True, input_args="--model-name YOLOv4 --load-id 300".split())
  args.batch_size = 32
  args.path_cp = Path("/home/yy/Coding/GitHub/KataCV/logs/YOLOv4-checkpoints")
  # args.path_cp = Path("/home/wty/Coding/GitHub/KataCV/logs/YOLOv4-checkpoints")

  from katacv.yolov4.yolov4_model import get_yolov4_state
  state = get_yolov4_state(args)

  from katacv.utils.model_weights import load_weights
  state = load_weights(state, args)

  from katacv.utils.coco.build_dataset import DatasetBuilder
  # from katacv.utils.pascal.build_dataset import DatasetBuilder
  ds_builder = DatasetBuilder(args)
  train_ds = ds_builder.get_dataset(subset='train')
  val_ds = ds_builder.get_dataset(subset='val')
  # sample_ds = ds_builder.get_dataset(subset='sample8')

  predictor = Predictor(args, state)

  bar = tqdm(enumerate(val_ds), total=len(val_ds))
  # bar = tqdm(enumerate(train_ds), total=len(train_ds))
  for i, (x, tbox, tnum) in bar:
    x, tbox, tnum = x.numpy(), tbox.numpy(), tnum.numpy()
    predictor.update(x, tbox, tnum)
    # for j in range(args.batch_size):
    #   idx = i * args.batch_size + j
    #   pbox = predictor.pbox[idx]
    #   show_bbox(x[j], pbox, args.path_dataset.name)
    #   print("pbox:", pbox)
    #   print("tbox:", tbox[0][:tnum[0]])
    #   show_bbox(x[j], tbox[0][:tnum[0]], args.path_dataset.name)
    result = predictor.p_r_ap50_ap75_map()
    metrics = {
      'P@50': result[0],
      'R@50': result[1],
      'AP@50': result[2],
      'AP@75': result[3],
      'mAP': result[4],
    }
    bar.set_description(' '.join(f"{key}={value:.4f}" for key, value in metrics.items()))

if __name__ == '__main__':
  # path_tg = Path("/home/wty/Coding/GitHub/Object-Detection-Metrics-master/groundtruths")
  # path_pred = Path("/home/wty/Coding/GitHub/Object-Detection-Metrics-master/detections")
  # metric_from_file(path_tg, path_pred)

  metric_from_model()
