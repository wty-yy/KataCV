import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import jax, jax.numpy as jnp

def plot_box(ax: plt.Axes, image_shape: tuple[int], box_params: tuple[float] | np.ndarray, text="", fontsize=8, box_color='red'):
    """
    (Matplotlib) Plot the bounding box with `box_params` in `ax`.
    ### Params
    `box_params`: (x, y, w, h) is proportion of the image, so we need `image_shape`
    - (x, y) is the center of the box.
    - (w, h) is the width and height of the box.

    `text`: The text display in the upper left of the bounding box.
    """
    if type(box_params) != np.ndarray: box_params = np.array(box_params)
    assert(box_params.size == 4)
    params, shape = box_params, image_shape
    x_min = int(shape[1]*(params[0]-params[2]/2))
    y_min = int(shape[0]*(params[1]-params[3]/2))
    w = int(shape[1] * params[2])
    h = int(shape[0] * params[3])
    rect = patches.Rectangle((x_min, y_min), w, h, linewidth=2, edgecolor=box_color, facecolor='none')
    ax.scatter(int(shape[1] * params[0]), int(shape[0] * params[1]), color='yellow', s=50)
    ax.add_patch(rect)
    bbox_props = dict(boxstyle="round, pad=0.2", edgecolor=box_color, facecolor=box_color)
    if len(text) != 0:
        ax.text(x_min+2, y_min, text, color='white', backgroundcolor=box_color, va='bottom', ha='left', fontsize=fontsize, bbox=bbox_props)

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
def plot_box_PIL(
        image:Image.Image, box_params: tuple | np.ndarray,
        text="", fontsize=14, box_color='red',
        fontpath="../fonts/Consolas.ttf",
        format='yolo',
        draw_center_point=False,
    ):
    """
    (PIL) Plot the bounding box with `box_params` in `image`.

    Args:
        text: The text display in the upper left of the bounding box.
        fontpath: The relative to path `katacv/utils/detection`.
        format: Support three common bounding boxes type: \
            - `yolo` (proportion): `(x, y, w, h)`, `(x, y)` is the center of the bbox, \
            and `(w, h)` is the width and height of the bbox. \
            - `coco` (pixel): `(x, y, w, h)`, `(x, y)` is the left top of the bbox, \
            and `(w, h)` is the width and height of the bbox. \
            - `voc` (pixel): `(x1, y1, x2, y2)`, `(x1, y1)` is the left top of the bbox, \
            and `(x2, y2)` is the rigth bottom of the bbox.
    """
    fontpath = str(Path(__file__).parent.joinpath(fontpath).resolve())
    draw = ImageDraw.Draw(image)
    params, shape = np.array(box_params), image.size
    if np.max(params) <= 1.0:
        params[0] *= shape[0]
        params[2] *= shape[0]
        params[1] *= shape[1]
        params[3] *= shape[1]
    if format.lower() == 'yolo':
        x_min = int(params[0] - params[2] / 2)
        y_min = int(params[1] - params[3] / 2)
        w = int(params[2])
        h = int(params[3])
    elif format.lower() == 'coco':
        x_min = int(params[0])
        y_min = int(params[1])
        w = int(params[2])
        h = int(params[3])
    elif format.lower() == 'voc':
        x_min = int(params[0])
        y_min = int(params[1])
        w = int(params[2] - params[0])
        h = int(params[3] - params[1])
    if type(box_color) == str and box_color == 'red': box_color = (255, 0, 0)
    draw.rectangle([x_min, y_min, x_min+w, y_min+h], outline=box_color, width=2)

    font_color = (255,255,255)  # white
    font = ImageFont.truetype(fontpath, fontsize)
    w_text, h_text= font.getsize(text)  # pip install Pillow==9.5.0
    x_text = x_min
    y_text = y_min - h_text if y_min > h_text else y_min
    draw.rounded_rectangle([x_text, y_text, x_text+w_text, y_text+h_text], radius=1.5, fill=box_color)
    draw.text((x_text, y_text), text, fill=font_color, font=font)
    if draw_center_point:
        draw.rounded_rectangle([x_min+w/2-2,y_min+h/2-2,x_min+w/2+2,y_min+h/2+2], radius=1.5, fill=(255,0,0))
    return image

def plot_cells(ax: plt.Axes, image_shape: tuple[int], S: int):
    """
    Draw the cells division on the image ax.
    params::S: Split the image into SxS cells.
    """
    x_size, y_size = int(image_shape[1] / S), int(image_shape[0] / S)
    for i in range(1,S):
        ax.plot([i*x_size, i*x_size], [0, image_shape[0]], c='b', ls='--')
        ax.plot([0, image_shape[1]], [i*y_size, i*y_size], c='b', ls='--')

def slice_by_idxs(a: jax.Array, idxs: jax.Array, follow_nums: int):
    """
    Slice `a` by `idxs`,
    require `a[...,0].size == idxs.size` and `max(idxs.size)+follow_nums <= a.shape[-1]`

    Example: take the best boxes in YOLO 
    """
    reduce_size = a[...,0].size
    b = a.reshape(-1, a.shape[-1])
    idx1 = jnp.arange(reduce_size).reshape(reduce_size,1)+jnp.zeros(follow_nums, dtype='int32')
    idx2 = idxs.reshape(-1, 1)+jnp.arange(follow_nums)
    result = b[idx1, idx2].reshape([*a.shape[:-1], follow_nums])
    return result

from functools import partial
BoxType = jax.Array
@partial(jax.jit, static_argnums=[2,3,4,5])
def iou(
    box1: BoxType, box2: BoxType,
    format: str = 'iou',
    scale: list | jax.Array = None,
    keepdim: bool = False, EPS: float = 1e-6
):
    """
    (JAX) Calculate the intersection over union for box1[i] and box2[i] (YOLO format).
    @params::box1, box2: let `shape` be the shape of box1 and box2.
        if shape[-1] == 4, then the last dim is `(x,y,w,h)`.
        if shape[-1] == 2, then the last dim is `(w,h)`.
        (x, y): the center of the box.
        (w, h): the width and the height of the box.

        shape[0] are the same or
        allow one of shape[0] == 1, it will implement boardcast.
    @params::format: `iou` or `diou` or `ciou`
    @return::IOU of box1 and box2, `shape=(N,)` when `keepdim=False`
    """
    assert(format in ['iou', 'diou', 'ciou'])
    if box1.ndim == 1: box1 = box1.reshape(1,-1)
    if box2.ndim == 1: box2 = box2.reshape(1,-1)
    assert(box1.shape[-1] == box2.shape[-1])
    if box1.shape[-1] == 2:
        box1 = jnp.pad(box1, ((0,0), (2,0)))
        box2 = jnp.pad(box2, ((0,0), (2,0)))
    assert(box1.shape[-1] == 4)

    if scale is not None:
        if type(scale) == list: scale = jnp.array(scale)
        box1 *= scale; box2 *= scale
    min1, min2 = box1[...,0:2]-box1[...,2:4]/2, box2[...,0:2]-box2[...,2:4]/2
    max1, max2 = box1[...,0:2]+box1[...,2:4]/2, box2[...,0:2]+box2[...,2:4]/2
    inter_h = (jnp.minimum(max1[...,0],max2[...,0]) - jnp.maximum(min1[...,0],min2[...,0])).clip(0.0)
    inter_w = (jnp.minimum(max1[...,1],max2[...,1]) - jnp.maximum(min1[...,1],min2[...,1])).clip(0.0)
    inter_size = inter_h * inter_w
    size1, size2 = jnp.prod(max1-min1, axis=-1), jnp.prod(max2-min2, axis=-1)
    union_size = size1 + size2 - inter_size
    result_iou = inter_size / (union_size + EPS)  # IOU
    if format == 'iou':
        ret = result_iou
        if keepdim: ret = ret[...,None]
        return ret

    outer_h = jnp.maximum(max1[...,0],max2[...,0]) - jnp.minimum(min1[...,0],min2[...,0])
    outer_w = jnp.maximum(max1[...,1],max2[...,1]) - jnp.minimum(min1[...,1],min2[...,1])
    center_dist = ((box1[...,:2]-box2[...,:2])**2).sum(-1)
    # diagonal_dist = jax.lax.stop_gradient(outer_h**2 + outer_w**2)  # Update (2023/12/29): Stop diagonal gradient.
    diagonal_dist = outer_h**2 + outer_w**2  # Update (2023/12/29): Stop diagonal gradient.
    result_diou = result_iou - center_dist / (diagonal_dist + EPS)  # DIOU
    if format == 'diou':
        ret = result_diou
        if keepdim: ret = ret[...,None]
        return ret

    v = 4 / (jnp.pi ** 2) * (
        jnp.arctan(box1[...,2]/(box1[...,3]+EPS)) -
        jnp.arctan(box2[...,2]/(box2[...,3]+EPS))
    ) ** 2
    alpha = jax.lax.stop_gradient(v / (1 - result_iou + v))  # must use stop gradient !
    S = jax.lax.stop_gradient(result_iou >= 0.5)  # https://arxiv.org/pdf/2005.03572.pdf
    # S = 1
    result_ciou = result_diou - S * alpha * v
    if format == 'ciou': ret = result_ciou
    if keepdim: ret = ret[...,None]
    return ret

@partial(jax.jit, static_argnums=[2])
def iou_multiply(boxes1, boxes2, format='iou'):
    """
    Return the IOU after pairwise combination `boxes1` and `boxes2`
    @params::`boxes1.shape=(N,4), boxes2.shape=(M,4)`.
    @params::`format` = `iou` or `diou` or `ciou`
    @return::`shape=(N,M)`, the `(i,j)` element of the return matrix is the IOU of `boxes1[i]` and `boxes2[j]`.
    """
    _, result = jax.lax.scan(
        f=lambda carry, x: (carry, iou(x, carry, format)),
        init=boxes2,
        xs=boxes1
    )
    return result

@partial(jax.jit, static_argnums=[3,4])
def nms_boxes_and_mask_old(boxes, iou_threshold=0.3, conf_threshold=0.2, max_num_box=100, B=3, iou_format='iou'):
    """
    (JAX) Non-Maximum Suppression with `iou_threshold` and `conf_threhold`.
    Make sure to keep the matrices in same size, we must give the maximum number of bounding boxes in an image.
    ### Params
    - `boxes.shape=(N,5)`, last dim is `(c,x,y,w,h,cls)`
    - `iou_threshold`: The threshold of maximum iou in NMS.
    - `conf_threshold`: The threshold of minimum confidence in NMS.
    - `max_num_box`: The maximum number of bounding boxes in an image.
    - `B`: The maximum number of bounding boxes in one scale.
    ### Return
    - `boxes`: The confidence top `max_num_box * B`.
    - `mask`: The mask of the boxes after NMS.
    ### Useage
    `boxes, mask = nms_boxes_and_mask(boxes, iou_threshold, conf_threshold)`

    `boxes = boxes[mask]`
    """
    M = max_num_box * B
    sort_idxs = jnp.argsort(boxes[:,0])[::-1][:M]  # only consider the first `M`
    boxes = boxes[sort_idxs]
    ious = iou_multiply(boxes[:,1:5], boxes[:,1:5], format=iou_format)
    mask = (boxes[:,0] > conf_threshold) & (~jnp.diagonal(jnp.tri(M,k=-1) @ (ious > iou_threshold)).astype('bool'))
    return boxes, mask

@partial(jax.jit, static_argnums=[3,4,5])
def nms(box, iou_threshold=0.3, conf_threshold=0.2, nms_multi=30, max_num_box=100, iou_format='iou'):
  """
  Compute the predicted bounding boxes and the number of bounding boxes.
  
  Args:
    box: The predicted result by the model.  [shape=(N,6), elem=(x,y,w,h,conf,cls)]
    iou_threshold: The IOU threshold of NMS.
    conf_threshold: The confidence threshold of NMS.
    max_num_box: The maximum number of the bounding boxes.
    iou_format: The format of IOU is used in calculating IOU threshold.
    nms_multi: `max_num_box * num_multi` is the pre box number for iou calculation.
  
  Return:
    box: The bounding boxes after NMS.  [shape=(max_num_box, 6)]
    pnum: The number of the predicted bounding boxes. [int]
  """
  M = min(max_num_box * nms_multi, box.shape[0])  # BUG FIX: The M must bigger than max_num_box, since iou threshold will remove many boxes beside.
  sort_idxs = jnp.argsort(-box[:,4])[:M]  # only consider the first `max_num_box`
  box = box[sort_idxs]
  ious = iou_multiply(box[:,:4], box[:,:4], format=iou_format)
  mask = (box[:,4] > conf_threshold) & (~jnp.diagonal(jnp.tri(M,k=-1) @ (ious > iou_threshold)).astype('bool'))
  idx = jnp.argwhere(mask, size=max_num_box, fill_value=-1)[:,0]  # nonzeros
  dbox = box[idx]
  pnum = (idx != -1).sum()
  return dbox, pnum

@partial(jax.jit, static_argnums=[3,4,5])
def nms_bad(pbox, iou_thre=0.65, conf_thre=0.001, max_box=300, max_size=3000, iou_format='iou'):
  """
  (JAX) Compute the predicted bounding boxes and the number of bounding boxes. (too slow)
  
  Args:
    box: The predicted result by the model.  [shape=(N,6), elem=(x,y,w,h,conf,cls)]
    iou_thre: The IOU threshold of NMS.
    conf_thre: The confidence threshold of NMS.
    max_box: The maximum return number of the bounding boxes.
    max_size: The maximum size of considering the predict boxes.
    iou_format: The format of IOU is used in calculating IOU threshold.
  
  Return:
    box: The bounding boxes after NMS.  [shape=(max_box, 6)]
    pnum: The number of the predicted bounding boxes. [int]
  """
  pbox = pbox[jnp.argsort(-pbox[:,4])]
  if pbox.shape[0] > max_size: pbox = pbox[:max_size]
  keep = pbox[:,4] > conf_thre
  identity = lambda x: x
  def loop_i_fn(i, keep):
    def loop_j_fn(j, keep):
      def modify(keep):
        return keep.at[j].set(False)
      keep = jax.lax.cond(
        keep[j],
        lambda k: jax.lax.cond(
            iou(pbox[i][:4], pbox[j][:4], format=iou_format)[0] > iou_thre,
            modify, identity, k
        ),
        identity,
        keep)
      return keep
    keep = jax.lax.cond(
      keep[i],
      lambda k: jax.lax.fori_loop(i+1, pbox.shape[0], loop_j_fn, k),
      identity,
      keep)
    return keep
  keep = jax.lax.fori_loop(0, pbox.shape[0], loop_i_fn, keep)
  idx = jnp.argwhere(keep, size=max_box, fill_value=-1)[:,0]
  pbox = pbox[idx]
  pnum = (idx != -1).sum()
  return pbox, pnum

# def nms(boxes, iou_threshold=0.3, conf_threshold=0.2):
#     """
#     (Numpy) Calculate the Non-Maximum Suppresion for boxes between the classes in **one sample**.
#     @params::`boxes.shape=(M,6)` and last dim is `(c,x,y,w,h,cls)`.
#     @return::the boxes after NMS.
#     """
#     boxes = boxes[boxes[:,0]>conf_threshold]
#     conf_sorted_idxs = np.argsort(boxes[:,0])[::-1]
#     boxes = boxes[conf_sorted_idxs]
#     M = boxes.shape[0]
#     used = np.zeros(M, dtype='bool')
#     boxes_nms = np.zeros_like(boxes)
#     tot = 0
#     for i in range(M):
#         if used[i]: continue
#         boxes_nms[tot] = boxes[i]; tot += 1
#         for j in np.where((~used) & (np.arange(M)>i) & (boxes[:,5]==boxes[i,5]))[0]:
#             if iou(boxes[i,1:5], boxes[j,1:5])[0] > iou_threshold:
#                 used[j] = True
#     boxes_nms = boxes_nms[:tot]
#     return boxes_nms

def nms_old(boxes, iou_threshold=0.3, conf_threshold=0.2):
    """
    Calculate the Non-Maximum Suppresion for boxes between the classes in **one sample**.
    @params::`boxes.shape=(M,6)` and last dim is `(c,x,y,w,h,cls)`.
    @return::the boxes after NMS.
    """
    if type(boxes) != list:
        boxes = list(boxes)
    boxes = [box for box in boxes if box[0] > conf_threshold]
    boxes = sorted(boxes, key=lambda x: x[0], reverse=True)
    boxes_after_nms = []

    while boxes:
        chosen_box = boxes.pop(0)
        boxes = [
            box for box in boxes
            if box[5] != chosen_box[5]
            or iou(chosen_box[1:5], box[1:5])[0] < iou_threshold
        ]
        boxes_after_nms.append(chosen_box)
        
    return np.array(boxes_after_nms)

def mAP(boxes, target_boxes, iou_threshold=0.5):
    """
    Calculate the mAP (AP: area under PR curve) of the boxes and the target_boxes with the iou threshold.
    @params::boxes.shape=(N,6) and last dim is (c,x,y,w,h,cls).
    @params::target_boxes.shape=(N,6) and last dim is (c,x,y,w,h,cls).
    """
    classes = jnp.unique(target_boxes[:,5])
    APs = 0
    for cls in classes:
        p, r = 1.0, 0.0  # update
        if (boxes[:,5]==cls).sum() == 0: continue
        box1 = boxes[boxes[:,5]==cls]
        sorted_idxs = jnp.argsort(box1[:,0])[::-1]  # use argsort at conf, don't use sort!
        box1 = box1[sorted_idxs]
        box2 = target_boxes[target_boxes[:,5]==cls]
        TP, FP, FN, AP = 0, 0, box2.shape[0], 0
        used = [False for _ in range(box2.shape[0])]
        for i in range(box1.shape[0]):
            match = False
            for j in range(box2.shape[0]):
                if used[j] or iou(box1[i,1:5], box2[j,1:5])[0] <= iou_threshold: continue
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

def get_best_boxes_and_classes(cells, B, C):
    """
    (JAX)Get the best confidence boxes and classes in cells,
    and convert the `x, y` that relative cell to relative whole image.
    @param::cells `cells.shape=(N,S,S,C+5*B)`
    @return `boxes.shape=(N,SxS,6)`, the last dim's mean: `(c,x,y,w,h,cls)`
    @speed up::`func = jax.jit(get_best_boxes_and_classes, static_argnums=[1,2,3])`
    """
    N = cells.shape[0]
    cls = jnp.argmax(cells[...,:C], -1)                 # (N,S,S)
    probas = slice_by_idxs(cells, cls, 1)               # (N,S,S,1)
    confs = cells[..., C+5*jnp.arange(B)]               # (N,S,S,B)
    best_idxs = jnp.argmax(confs, -1)                   # (N,S,S)
    b = slice_by_idxs(cells, C+best_idxs*5, 5)          # (N,S,S,5)
    b = b.at[...,1:3].set(cvt_coord_cell2image(b[...,1:3]))  # set again
    bc = b[..., 0] * probas[..., 0]                     # (N,S,S)
    bx, by, bw, bh = [b[...,i+1] for i in range(4)]     # (N,S,S)
    return jnp.stack([bc,bx,by,bw,bh,cls], -1).reshape(N,-1,6)  # (N,SxS,6)

@jax.jit
def cvt_coord_cell2image(xy):
    """
    (JAX)Convert xy coordinates relative cell to relative the whole image.
    @params::xy `shape=(None,S,S,2)`, where `S` is the splite size of the cells.
    """
    assert(xy.shape[-1] == 2 and xy.shape[-2] == xy.shape[-3])
    origin_shape, W, H = xy.shape, xy.shape[-2], xy.shape[-3]
    if xy.ndim == 3: xy = xy.reshape(-1, H, W, 2)
    dx, dy = [jnp.repeat(x[None,...], xy.shape[0], 0) for x in jnp.meshgrid(jnp.arange(H), jnp.arange(W))]
    return jnp.stack([(xy[...,0]+dx)/W, (xy[...,1]+dy)/H], -1).reshape(origin_shape)

def cvt_one_yolov1_label2boxes(label, C):
    assert(label.ndim == 3)
    if type(label) != np.ndarray:
        label = np.array(label)
    label[...,C+1:C+3] = cvt_coord_cell2image(label[...,C+1:C+3])
    label = label[label[...,C] != 0]
    box = jnp.concatenate([label[...,C:],jnp.argmax(label[...,:C],-1,keepdims=True)],-1).reshape(-1,6)
    return box

def cvt_one_yolov3_label2boxes(label):
    """
    (Numpy) Convert one YOLOv3 label to origin bboxes (relative to the whole image),
    assume the labels output of the dataset is `labels`, 
    the bboxes in first example can be call by `cvt_one_yolov3_label2boxes(labels[0][0])`
    - label.shape=(S,S,B,6), elements: (c,x,y,w,h,cls)
    """
    if type(label) != np.ndarray:
        label = np.array(label)
    boxes = []
    for i in range(label.shape[2]):
        box = label[:,:,i,:]  # (S,S,5+C)
        box[...,1:3] = cvt_coord_cell2image(box[...,1:3])
        box[...,3:5] /= box.shape[0]
        box = box[box[...,0] == 1]
        if box.size != 0:
            boxes.append(box)
    return np.concatenate(boxes, 0)

def get_box_colors(n):
    cmap = plt.cm.brg
    step = cmap.N // n
    colors = cmap([i for i in range(0, cmap.N, step)])
    colors = (colors[:, :3] * 255).astype(int)
    colors = [tuple(color) for color in colors]
    return colors

def build_label2colors(labels):
    labels = np.unique(labels).astype(np.int32)
    colors = get_box_colors(len(labels))
    return dict(zip(labels, colors))

if __name__ == '__main__':
    a = jnp.array([1,1,2,2])
    b1 = jnp.array([0,0,1,1])
    b2 = jnp.array([0,0,10,1])
    b3 = jnp.array([10,0,2,2])
    b4 = jnp.array([0,2,2,2])
    aa = jnp.array([a,a,a,a])
    bb = jnp.array([b1,b2,b3,b4])
    print(iou(aa, bb, keepdim=True))
    print(iou_multiply(aa, bb, format='iou'))
    print(iou(aa, bb, 'diou', keepdim=True))
    print(iou_multiply(aa, bb, format='diou'))
    print(iou(aa, bb, 'ciou', keepdim=True))
    print(iou_multiply(aa, bb, format='ciou'))
    c = jnp.array([1,2])
    d1 = jnp.array([3,4])
    d2 = jnp.array([3,3])
    cc = jnp.array([c,c])
    dd = jnp.array([d1,d2])
    print(iou(cc, dd))
    e = jnp.array([0,0,0,0])
    print(iou(e, e, format='ciou'))
