import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import jax, jax.numpy as jnp

def plot_box(ax: plt.Axes, image_shape: tuple[int], box_params: tuple[float] | np.ndarray, text=""):
    """
    params::box_params: (x, y, w, h) is proportion of the image, so we need `image_shape`
        - (x, y) is the center of the box.
        - (w, h) is the width and height of the box.
    params::text: The text display in the upper left of the bounding box.
    """
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
        ax.text(x_min+2, y_min, text, color='white', backgroundcolor='red', va='bottom', ha='left', fontsize=8, bbox=bbox_props)

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
    slice `a` by `idxs`,
    require `a[...,0].size == idxs.size` and `max(idxs.size)+follow_nums <= a.shape[-1]`
    example: use it to take the best boxes in YOLO 
    """
    reduce_size = a[...,0].size
    b = a.reshape(-1, a.shape[-1])
    idx1 = jnp.arange(reduce_size).reshape(reduce_size,1)+jnp.zeros(follow_nums, dtype='int32')
    idx2 = idxs.reshape(-1, 1)+jnp.arange(follow_nums)
    result = b[idx1, idx2].reshape([*a.shape[:-1], follow_nums])
    return result

BoxType = jax.Array
def iou(box1: BoxType, box2: BoxType, EPS: float = 1e-6, keepdim: bool = False):
    """
    Calculate the intersection over union for box1 and box2.
    params::box1, box2 shapes are (N,5), where the last dim x,y,w,h under the **same scale**:
        (x, y): the center of the box.
        (w, h): the width and the height of the box.
    return::IOU of box1 and box2, shape=(N)
    """
    min1, min2 = box1[...,0:2]-box1[...,2:4]/2, box2[...,0:2]-box2[...,2:4]/2
    max1, max2 = box1[...,0:2]+box1[...,2:4]/2, box2[...,0:2]+box2[...,2:4]/2
    inter_w = jnp.minimum(max1[...,0],max2[...,0]) - jnp.maximum(min1[...,0],min2[...,0])
    inter_h = jnp.minimum(max1[...,1],max2[...,1]) - jnp.maximum(min1[...,1],min2[...,1])
    inter_size = jnp.where((inter_w<=0)|(inter_h<=0), 0, inter_w*inter_h)
    size1, size2 = jnp.prod(max1-min1, axis=-1), jnp.prod(max2-min2, axis=-1)
    union_size = size1 + size2 - inter_size
    ret = inter_size / (union_size + EPS)
    if keepdim: ret = ret[...,jnp.newaxis]
    return ret

def mAP(boxes, target_boxes, iou_threshold):
    pass

def coco_mAP(boxes, target_boxes):
    pass

if __name__ == '__main__':
    a = jnp.array([1,1,2,2]).reshape(1,-1)
    b1 = jnp.array([0,0,1,1]).reshape(1,-1)
    b2 = jnp.array([10,0,2,2]).reshape(1,-1)
    b3 = jnp.array([1,1,1,1]).reshape(1,-1)
    b4 = jnp.array([0,2,2,2]).reshape(1,-1)
    aa = jnp.concatenate([a,a,a,a], axis=0)
    bb = jnp.concatenate([b1,b2,b3,b4], axis=0)
    print(iou(aa, bb, keepdim=True))