from katacv.utils.related_pkgs.utility import *
from katacv.utils.related_pkgs.jax_flax_optax_orbax import *
from katacv.utils.detection import iou
from katacv.yolov5.parser import YOLOv5Args

def BCE(logits, y, mask):
  return -(mask * (
    y * jax.nn.log_sigmoid(logits) +
    (1-y) * jax.nn.log_sigmoid(-logits)
  )).mean()

def CIOU(box1, box2, mask):
  fn = partial(iou, format='ciou')
  return (mask * (1 - jax.vmap(fn, (0, 0))(box1, box2))).mean()

class ComputeLoss:
  def __init__(self, args: YOLOv5Args):
    self.batch_size = args.batch_size
    self.anchors = args.anchors
    self.nc = args.num_classes
    self.coef_box = args.coef_box
    self.coef_obj = args.coef_obj
    self.coef_cls = args.coef_cls
    self.balance_obj = [4.0, 1.0, 4.0]
    self.aspect_ratio_thre = 4.0
    self.offset = jnp.array(
      [(0, 0), (-1, 0), (1, 0), (0, 1), (0, -1)], dtype=jnp.float32
    ) * 0.5

  @partial(jax.jit, static_argnums=0)
  def train_step(
    self, state: train_state.TrainState,
    x: jnp.ndarray, box: jnp.ndarray,
    nb: jnp.ndarray, train: bool
  ):
    """
    Args:
      state: Flax TrainState
      x: Input images. [shape=(N,H,W,C)]
      box: Target boxes. [shape=(N,M,5)]
      nb: Number of boxes. [shape=(N,)]
      train: Update state if train.
    """
    def single_loss_fn(p, t, anchors):
      """
      Args:
        p (logits): [shape=(N,3,H,W,5+nc)]
        t (target): [shape=(N,3,H,W,6)]
        anchors: [shape=(3,2)]
      """
      mask = t[..., 4] == 1  # positive mask
      xy = (jax.nn.sigmoid(p[...,:2]) - 0.5) * 2.0 + 0.5
      wh = (jax.nn.sigmoid(p[2:4]) * 2) ** 2 * anchors[1,3,1,1,2]
      ious = iou(jnp.concatenate([xy, wh], -1), t[..., :4], format='ciou')
      lbox = (mask * (1 - ious)).mean()
      lobj = (
        BCE(p[..., 4], jnp.clip(ious, 0), mask) +  # positive
        BCE(p[..., 4], 0, 1-mask)  # negetive
      )
      hot = jax.nn.one_hot(t[..., 5], self.nc)
      lcls = BCE(p[..., 5:], hot, mask)
      return lbox, lobj, lcls
    
    def loss_fn(params):
      logits, updates = state.apply_fn(
        {'params': params, 'batch_stats': state.batch_stats},
        x, train=True, mutable=['batch_stats']
      )
      targets = jax.vmap(self.build_target)(logits, box, nb)
      lbox, lobj, lcls = 0, 0, 0
      for i in range(3):
        losses = single_loss_fn(logits[i], targets[i], self.anchors[i])
        lbox += losses[0]
        lobj += losses[1] * self.balance_obj[i]
        lcls += losses[2]
      lbox *= self.coef_box
      lobj *= self.coef_obj
      lcls *= self.coef_cls
      loss = self.batch_size * (lbox + lobj + lcls)
      return loss, (updates, lbox, lobj, lcls)
    if train:
      (loss, (updates, *metrics)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
      state = state.apply_gradients(grads=grads)
      state = state.replace(batch_stats=updates['batch_stats'])
    else:
      loss, (_, *metrics) = loss_fn(state.params)
    return state, (loss, *metrics)

  @partial(jax.jit, static_argnums=0)
  def build_target(self, p: List[jnp.ndarray], box: jnp.ndarray, nb: int):
    """
    Build target for one sample.
    Args:
      p (logits): list[shape=(3,Hi,Wi,5+nc)], i=0,1,2, \
        [elem: (x,y,w,h,conf,*prob)]
      box: Target boxes with YOLO format. [shape=(M,5)]
      nb: Number of the target box.
    Return:
      target: Target for `p` cell format. \
        list[shape=(3,Hi,Wi,6)], i=0,1,2, [elem: (x,y,w,h,conf,cls)]
    """
    target = [jnp.zeros((*p[i].shape[:3],6)) for i in range(3)]
    def loop_i_fn(i, target):  # box[i]
      b, cls = box[i, :4], box[i, 4]
      rate = b[None,None,2:4] / self.anchors  # anchors.shape=(3,3,2)
      flag = jnp.maximum(rate, 1.0 / rate).max(-1) < self.aspect_ratio_thre  # shape=(3,3)

      def update_fn(value):
        t, k, c, bc = value
        t = t.at[k,c[1],c[0]].set(jnp.r_[bc, 1, cls])
        return t

      for j in range(3):  # diff scale
        s = 2 ** (j+3)
        cs = (self.offset + b[:2] / s).astype(jnp.int32)  # center in cells, shape=(5,2)
        for c in cs:  # add target to near cell
          for k in range(3):  # diff anchor in current scale
            bc = jnp.r_[b[:2]/s - c.astype(jnp.float32), b[2:4]]
            target[j] = jax.lax.cond(
              flag[j,k], update_fn, lambda x: x[0], (target[j], k, c, bc)
            )
      return target
    target = jax.lax.fori_loop(0, nb, loop_i_fn, target)
    return target

def cell2pixel(xy, scale):
  assert xy.shape[-1] == 2
  h, w = xy.shape[-3:-1]
  if xy.ndim == 3: xy.reshape(-1, h, w, 2)
  dx, dy = [jnp.repeat(x[None,...], xy.shape[0], 0) for x in jnp.meshgrid(jnp.arange(h), jnp.arange(w))]
  return jnp.stack([(xy[...,0]+dx)*scale, (xy[...,1]+dy)*scale], -1)

def target_debug(x, target):
  from katacv.utils.yolo.utils import show_box
  from PIL import ImageDraw, Image
  import numpy as np
  image = Image.fromarray((x*255).astype(np.uint8))
  for i in range(3):
    t = target[i]
    s = 2**(i+3)
    print(f"Scale: {s}")
    xy = cell2pixel(t[...,:2], scale=s)
    for j in range(3):
      idxs = np.transpose((t[j,...,4]).nonzero())
      print(idxs)
      # idxs = ((9,2),(10,2),(10,3))
      colors = ((255,0,0), (0,255,0), (0,0,255))
      for k, (x, y) in enumerate(idxs):
        b = jnp.concatenate([xy[j,x,y,:], t[j,x,y,2:4], t[j,x,y,5:6]], -1)[None,...]
        image = show_box(image, b, verbose=False)
        draw = ImageDraw.Draw(image)
        draw.rectangle((y*s,x*s,(y+1)*s,(x+1)*s), fill=colors[k%3])
    break
  image.show()

if __name__ == '__main__':
  from katacv.yolov5.parser import get_args_and_writer
  args = get_args_and_writer(no_writer=True)
  args.batch_size = 2
  from katacv.utils.yolo.build_dataset import DatasetBuilder
  ds_builder = DatasetBuilder(args)
  ds = ds_builder.get_dataset(subset='val')
  p = [jnp.zeros((
    args.batch_size, 3,
    args.image_shape[0]//(2**i),
    args.image_shape[1]//(2**i),
    5+80
  )) for i in range(3, 6)]
  comput_loss = ComputeLoss(args)
  for x, box, nb in ds:
    x, box, nb = x.numpy() / 255, box.numpy(), nb.numpy()
    target = jax.device_get(jax.vmap(comput_loss.build_target)(p, box, nb))
    for i in range(args.batch_size):
      i = 0
      target_debug(x[i], [t[i] for t in target])
      break
    break

