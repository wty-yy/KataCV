from katacv.utils.related_pkgs.jax_flax_optax_orbax import *

@jax.jit
def logits2prob(logits):  # (..., (x,y,w,h,c,num_classes))
  conf = jax.nn.sigmoid(logits[...,4:5]) * jnp.max(jax.nn.softmax(logits[...,5:]), axis=-1, keepdims=True)
  cls = jnp.argmax(logits[...,5:], axis=-1, keepdims=True)
  return jnp.concatenate([logits[...,:4], conf, cls], axis=-1)

def logits2prob_from_list(pred_pixel):
  N = pred_pixel[0].shape[0]
  return jnp.concatenate([logits2prob(pred_pixel[i]).reshape(N, -1, 6) for i in range(3)], axis=1)

def logits2cell(logits: jax.Array):
  xy = (jax.nn.sigmoid(logits[...,:2]) - 0.5) * 2.0 + 0.5  # xy range: (-0.5, 1.5)
  wh = (jax.nn.sigmoid(logits[...,2:4])*2)**2  # wh range: (0, 4)
  # xy = jax.nn.sigmoid(logits[...,:2])
  # wh = jnp.exp(logits[...,2:4])
  return jnp.concatenate([xy, wh, logits[...,4:]], axis=-1)

from katacv.utils.detection import iou, iou_multiply
@partial(jax.jit, static_argnums=[3,4])
def nms_boxes_and_mask_old(boxes, iou_threshold=0.3, conf_threshold=0.2, max_num_box=100, iou_format='diou'):
  M = max_num_box
  sort_idxs = jnp.argsort(-boxes[:,4])[:M]  # only consider the first `max_num_box`
  boxes = boxes[sort_idxs]
  ious = iou_multiply(boxes[:,:4], boxes[:,:4], format=iou_format)
  mask = (boxes[:,4] > conf_threshold) & (~jnp.diagonal(jnp.tri(M,k=-1) @ (ious > iou_threshold)).astype('bool'))
  return boxes, mask

@partial(jax.jit, static_argnums=[3,4])
def nms(box, iou_threshold=0.3, conf_threshold=0.2, max_num_box=100, iou_format='diou'):
  """
  Compute the predicted bounding boxes and the number of bounding boxes.
  
  Args:
    box: The predicted result by the model.  [shape=(N,6), elem=(x,y,w,h,conf,cls)]
    iou_threshold: The IOU threshold of NMS.
    conf_threshold: The confidence threshold of NMS.
    max_num_box: The maximum number of the bounding boxes.
    iou_format: THe format of IOU is used in calculating IOU threshold.
  
  Return:
    box: The bounding boxes after NMS.  [shape=(max_num_box, 6)]
    pnum: The number of the predicted bounding boxes. [int]
  """
  M = max_num_box * 9  # BUG FIX: The M must bigger than max_num_box, since iou threshold will remove many boxes beside.
  sort_idxs = jnp.argsort(-box[:,4])[:M]  # only consider the first `max_num_box`
  box = box[sort_idxs]
  ious = iou_multiply(box[:,:4], box[:,:4], format=iou_format)
  mask = (box[:,4] > conf_threshold) & (~jnp.diagonal(jnp.tri(M,k=-1) @ (ious > iou_threshold)).astype('bool'))
  idx = jnp.argwhere(mask, size=max_num_box, fill_value=-1)[:,0]  # nonzeros
  dbox = box[idx]
  pnum = (idx != -1).sum()
  return dbox, pnum

def get_pred_bboxes(pred, iou_threshold=0.3, conf_threshold=0.2):
  ret = []
  for i in range(pred.shape[0]):
    pbox, pnum = jax.device_get(nms(pred[i], iou_threshold=iou_threshold, conf_threshold=conf_threshold))
    pbox = pbox[:pnum]
    ret.append(pbox)
  return ret

from PIL import Image
def show_bbox(image, bboxes, dataset='coco', show_image=True):
  # print(bboxes)
  from katacv.utils.detection import plot_box_PIL, build_label2colors
  if dataset.lower() == 'coco':
    from katacv.utils.coco.constant import label2name
  if dataset.lower() == 'pascal':
    from katacv.utils.pascal.constant import label2name
  image = Image.fromarray((image*255).astype('uint8'))
  if len(bboxes):
    label2color = build_label2colors(list(label2name.keys()))
    # label2color = build_label2colors(bboxes[:,5])
  for bbox in bboxes:
    if len(bbox) == 6:  # predicted bbox
      label = int(bbox[5])
      image = plot_box_PIL(image, bbox[:4], text=label2name[label]+f"{bbox[4]:.2f}", box_color=label2color[label], format='yolo')
    else:  # target bbox
      label = int(bbox[4])
      image = plot_box_PIL(image, bbox[:4], text=label2name[label], box_color=label2color[label], format='yolo')
    # print(label, label2name[label], label2color[label])
  if show_image:
    image.show()
  return image

def mAP(boxes, target_boxes, iou_threshold=0.5):
  """
  Calculate the mAP (AP: area under PR curve) of the boxes and the target_boxes with the iou threshold.
  @params::boxes.shape=(N,6) and last dim is (x,y,w,h,c,cls).
  @params::target_boxes.shape=(N,5) and last dim is (x,y,w,h,cls).
  """
  classes = jnp.unique(target_boxes[:,4])
  if classes.size == 0:
    return 1.0
  APs = 0
  for cls in classes:
    p, r = 1.0, 0.0  # update
    if (boxes[:,5]==cls).sum() == 0: continue
    box1 = boxes[boxes[:,5]==cls]
    sorted_idxs = jnp.argsort(box1[:,4])[::-1]  # use argsort at conf, don't use sort!
    box1 = box1[sorted_idxs]
    box2 = target_boxes[target_boxes[:,4]==cls]
    TP, FP, FN, AP = 0, 0, box2.shape[0], 0
    used = [False for _ in range(box2.shape[0])]
    for i in range(box1.shape[0]):
      match = False
      for j in range(box2.shape[0]):
        if used[j] or iou(box1[i,:4], box2[j,:4])[0] <= iou_threshold: continue
        TP += 1; FN -= 1; used[j] = True; match = True
        break
      if not match: FP += 1
      last_p, p, last_r, r = p, TP/(TP+FP), r, TP/(TP+FN)
      AP += (last_p + p) * (r - last_r) / 2
    APs += AP
  return APs / classes.size

def coco_mAP(boxes, target_boxes):
  """
  Calculate the mAP with iou threshold [0.5,0.55,0.6,...,0.9,0.95]
  """
  ret = 0
  for iou_threshold in 0.5+jnp.arange(10)*0.05:
    ret += mAP(boxes, target_boxes, iou_threshold)
  return ret / 10

def calc_AP50_AP75_AP(pred_bboxes: list, bboxes, num_bboxes):
  AP50, AP75, AP = 0, 0, 0
  mAP_fn = lambda i, thre: mAP(
    boxes=pred_bboxes[i],
    target_boxes=bboxes[i][:num_bboxes[i]],
    iou_threshold=thre
  )
  coco_mAP_fn = lambda i: coco_mAP(
    boxes=pred_bboxes[i],
    target_boxes=bboxes[i][:num_bboxes[i]]
  )
  for i in range(len(pred_bboxes)):
    AP50 += (mAP_fn(i, 0.5) - AP50) / (i + 1)
    AP75 += (mAP_fn(i, 0.75) - AP75) / (i + 1)
    AP += (coco_mAP_fn(i) - AP) / (i + 1)
  return AP50, AP75, AP
