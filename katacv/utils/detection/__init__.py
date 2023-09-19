import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import jax, jax.numpy as jnp

def plot_box(ax: plt.Axes, image_shape: tuple[int], box_params: tuple[float] | np.ndarray, text="", fontsize=8):
    """
    params::box_params: (x, y, w, h) is proportion of the image, so we need `image_shape`
        - (x, y) is the center of the box.
        - (w, h) is the width and height of the box.
    params::text: The text display in the upper left of the bounding box.
    """
    assert(box_params.size == 4)
    params, shape = box_params, image_shape
    x_min = int(shape[1]*(params[0]-params[2]/2))
    y_min = int(shape[0]*(params[1]-params[3]/2))
    w = int(shape[1] * params[2])
    h = int(shape[0] * params[3])
    rect = patches.Rectangle((x_min, y_min), w, h, linewidth=2, edgecolor='r', facecolor='none')
    ax.scatter(int(shape[1] * params[0]), int(shape[0] * params[1]), color='yellow', s=50)
    ax.add_patch(rect)
    bbox_props = dict(boxstyle="round, pad=0.2", edgecolor='red', facecolor='red')
    if len(text) != 0:
        ax.text(x_min+2, y_min, text, color='white', backgroundcolor='red', va='bottom', ha='left', fontsize=fontsize, bbox=bbox_props)

def plot_cells(ax: plt.Axes, image_shape: tuple[int], S: int):
    """
    Draw the cells division on the image ax.
    params::S: Split the image into SxS cells.
    """
    x_size, y_size = int(image_shape[1] / S), int(image_shape[0] / S)
    for i in range(1,S):
        ax.plot([i*x_size, i*x_size], [0, image_shape[0]], c='b')
        ax.plot([0, image_shape[1]], [i*y_size, i*y_size], c='b')

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

BoxType = jax.Array
def iou(box1: BoxType, box2: BoxType, scale: list | jax.Array = None, keepdim: bool = False, EPS: float = 1e-6):
    """
    Calculate the intersection over union for box1 and box2.
    @params::box1, box2 shapes are (N,4), where the last dim x,y,w,h under the **same scale**:
        (x, y): the center of the box.
        (w, h): the width and the height of the box.
    @return::IOU of box1 and box2, shape=(N)
    """
    assert(box1.shape[-1] == box2.shape[-1])
    assert(box1.shape[-1] == 4)
    if box1.ndim == 1: box1 = box1.reshape(1,-1)
    if box2.ndim == 1: box2 = box2.reshape(1,-1)
    if scale is not None:
        if type(scale) == list: scale = jnp.array(scale)
        box1 *= scale; box2 *= scale
    min1, min2 = box1[...,0:2]-jnp.abs(box1[...,2:4])/2, box2[...,0:2]-jnp.abs(box2[...,2:4])/2
    max1, max2 = box1[...,0:2]+jnp.abs(box1[...,2:4])/2, box2[...,0:2]+jnp.abs(box2[...,2:4])/2
    inter_w = (jnp.minimum(max1[...,0],max2[...,0]) - jnp.maximum(min1[...,0],min2[...,0])).clip(0.0)
    inter_h = jnp.minimum(max1[...,1],max2[...,1]) - jnp.maximum(min1[...,1],min2[...,1]).clip(0.0)
    # inter_size = jnp.where((inter_w<=0)|(inter_h<=0), 0, inter_w*inter_h)
    inter_size = inter_w * inter_h
    size1, size2 = jnp.prod(max1-min1, axis=-1), jnp.prod(max2-min2, axis=-1)
    union_size = size1 + size2 - inter_size
    ret = inter_size / (union_size + EPS)
    if keepdim: ret = ret[...,jnp.newaxis]
    return ret

def nms(boxes, iou_threshold=0.3, conf_threshold=0.2):
    """
    (Numpy)Calculate the Non-Maximum Suppresion for boxes between the classes in **one sample**.
    @params::`boxes.shape=(M,6)` and last dim is `(c,x,y,w,h,cls)`.
    @return::the boxes after NMS.
    """
    boxes = boxes[boxes[:,0]>conf_threshold]
    conf_sorted_idxs = np.argsort(boxes[:,0])[::-1]
    boxes = boxes[conf_sorted_idxs]
    M = boxes.shape[0]
    used = np.zeros(M, dtype='bool')
    boxes_nms = np.zeros_like(boxes)
    tot = 0
    for i in range(M):
        if used[i]: continue
        boxes_nms[tot] = boxes[i]; tot += 1
        for j in np.where((~used) & (np.arange(M)>i) & (boxes[:,5]==boxes[i,5]))[0]:
            if jax.jit(iou)(boxes[i,1:5], boxes[j,1:5])[0] > iou_threshold:
                used[j] = True
    boxes_nms = boxes_nms[:tot]
    return boxes_nms

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

def nms_old(boxes, iou_threshold=0.5, conf_threshold=0.4):
    """
    Calculate the Non-Maximum Suppresion for boxes between the classes.
    params::boxes.shape=(N,6) and last dim is (c,x,y,w,h,cls).
    return::the boxes after NMS.
    """
    classes = boxes[:,5]
    uniq_classes = jnp.unique(classes)
    ret = []
    for cls in uniq_classes:
        now = []
        rank_boxes = boxes[classes == cls]
        rank_boxes = rank_boxes[jnp.argsort(rank_boxes[:,0])[::-1]]
        # rank_boxes = jnp.sort(, axis=1)[::-1,:]  # sort by confidence decrease
        for i in range(rank_boxes.shape[0]):
            box1 = rank_boxes[i,1:5]
            if rank_boxes[i,0] < conf_threshold: continue
            if box1[2] <= 0 or box1[3] <= 0:
                continue
            bad = False
            for item in now:
                box2 = item[1:5]
                # print(box1, box2)
                # print(iou(box1, box2)[0])
                if iou(box1, box2)[0] > iou_threshold:
                    bad = True; break
            if not bad:
                now.append(rank_boxes[i])
        ret += now
    return jnp.array(ret)

def mAP(boxes, target_boxes, iou_threshold=0.5):
    """
    Calculate the mAP of the boxes and the target_boxes with the iou threshold.
    params::boxes.shape=(N,6) and last dim is (c,x,y,w,h,cls).
    params::target_boxes.shape=(N,6) and last dim is (c,x,y,w,h,cls).
    """
    classes = jnp.unique(target_boxes[:,5])
    APs, min_p = 0, int(1e9)
    for cls in classes:
        r = 0  # update
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
            min_p, last_r, r = min(min_p, TP/(TP+FP)), r, TP/(TP+FN)  # update
            AP += min_p * (r - last_r)  # update
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

def cvt_coord_cell2image(xy):
    """
    (JAX)Convert xy coordinates relative cell to relative the whole image.
    @params::xy `shape=(None,S,S,2)`, where `S` is the splite size of the cells.
    """
    assert(xy.shape[-1] == 2 and xy.shape[-2] == xy.shape[-3])
    origin_shape, S = xy.shape, xy.shape[-2]
    if xy.ndim == 3: xy = xy.reshape(-1, S, S, 2)
    dx, dy = [jnp.repeat(x[None,...], xy.shape[0], 0) for x in jnp.meshgrid(jnp.arange(S), jnp.arange(S))]
    return jnp.stack([(xy[...,0]+dx)/S, (xy[...,1]+dy)/S], -1).reshape(origin_shape)

def cvt_one_label2boxes(label, C):
    assert(label.ndim == 3)
    if type(label) != np.ndarray:
        label = np.array(label)
    label[...,C+1:C+3] = cvt_coord_cell2image(label[...,C+1:C+3])
    label = label[label[...,C] != 0]
    box = jnp.concatenate([label[...,C:],jnp.argmax(label[...,:C],-1,keepdims=True)],-1).reshape(-1,6)
    return box

if __name__ == '__main__':
    a = jnp.array([1,1,2,2])
    b1 = jnp.array([0,0,1,1])
    b2 = jnp.array([10,0,2,2])
    b3 = jnp.array([1,1,1,1])
    b4 = jnp.array([0,2,2,2])
    aa = jnp.array([a,a,a,a])
    bb = jnp.array([b1,b2,b3,b4])
    print(iou(aa, bb, keepdim=True))
    print(iou(a, b1)[0])