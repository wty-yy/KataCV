from katacv.utils.related_pkgs.utility import *
from katacv.utils.related_pkgs.jax_flax_optax_orbax import *
from katacv.utils.detection import iou

@partial(jax.jit, static_argnames=['iou_ignore'])
def build_target(
    params: jax.Array,  # (M, 5), (x,y,w,h,label)
    num_bboxes: int,  # w, 
    # Shape: [(3, Hi, Wi, 5 + num_classes) for i in range(3)]
    pred_pixel: List[jax.Array],
    anchors: jax.Array,  # (3, 3, 2)
    iou_ignore: float = 0.5
  ) -> Tuple[List[jax.Array], List[jax.Array]]:
  """
  Return: Tuple -> (target, mask_noobj)
    target[i].shape = (3, Hi, Wi, 6)  # (x,y,w,h,c,label)
    mask_noobj[i].shape = (3, Hi, Wi)
    i = 0, 1, 2 means three diff scales 2 ** (i + 3)
  """
  params = jnp.array(params)
  target = [jnp.zeros((*pred_pixel[i].shape[:3],6)) for i in range(3)]
  mask = [jnp.zeros(pred_pixel[i].shape[:3], dtype=jnp.bool_) for i in range(3)]
  bboxes, labels = params[:, :4], params[:, 4].astype(jnp.int32)
  i = 0
  def loop_bbox_i_fn(i, value):
    target, mask = value
    bbox, label = bboxes[i], labels[i]
    # Update ignore examples
    for j in range(3):
      iou_pred = iou(
        bbox, pred_pixel[j][...,:4].reshape(-1,4)
      ).reshape(mask[j].shape)
      # label don't need correct, because the confidence is Pr(obj)
      # the last classification predict is Pr(class=C|obj)
      mask[j] = mask[j] | (iou_pred > iou_ignore)
    # Update best anchor target
    iou_anchors = iou(bbox[2:4], anchors.reshape(-1, 2), format='ciou')  # use ciou, add w/h ratio regular
    best_anchor = jnp.argmax(iou_anchors)
    j = (best_anchor / 3).astype(jnp.int32)
    k = best_anchor - 3 * j
    scale = 2 ** (j+3)
    bbox = bbox.at[:2].set(bbox[:2]/scale)
    cell = bbox[:2].astype(jnp.int32)
    bbox = bbox.at[:2].set(bbox[:2] - cell)
    bbox = bbox.at[2].set(bbox[2] / anchors[j,k,0])
    bbox = bbox.at[3].set(bbox[3] / anchors[j,k,1])
    # print(j, k, bbox, cell, anchors[j,k])
    def update_fn(value):
      target, mask = value
      target = target.at[k,cell[1],cell[0]].set(jnp.r_[bbox, 1, label])
      # target = target.at[k,cell[1],cell[0],5+label].set(1)
      mask = mask.at[k,cell[1],cell[0]].set(True)
      return target, mask
    for o in range(3):
      target[o], mask[o] = jax.lax.cond(
        o==j, update_fn, lambda x: x, (target[o], mask[o])
      )
    return target, mask
    # target[j] = target[j].at[k,cell[1],cell[0],:5].set(jnp.r_[bbox, 1])
    # target[j] = target[j].at[k,cell[1],cell[0],5+label].set(1)
    # mask[j] = mask[j].at[k,cell[1],cell[0]].set(True)
    # print(target[j][k,cell[1],cell[0]])
  target, mask = jax.lax.fori_loop(0, num_bboxes, loop_bbox_i_fn, (target, mask))
  for i in range(3):
    mask[i] = (mask[i] ^ True)[..., None]
  return target, mask

@jax.jit
def cell2pixel_coord(xy, scale: int):
  assert(xy.shape[-1] == 2 and xy.shape[-2] == xy.shape[-3])
  origin_shape, W, H = xy.shape, xy.shape[-2], xy.shape[-3]
  if xy.ndim == 3: xy = xy.reshape(-1, H, W, 2)
  dx, dy = [jnp.repeat(x[None,...], xy.shape[0], 0) for x in jnp.meshgrid(jnp.arange(H), jnp.arange(W))]
  return jnp.stack([(xy[...,0]+dx)*scale, (xy[...,1]+dy)*scale], -1).reshape(origin_shape)

@jax.jit
def cell2pixel(
  output: jax.Array,  # Shape: (3,Hi,Wi,5+num_classes)
  scale=int,  # 8 or 16 or 32
  anchors=jax.Array  # Shape: (3,2)
):
  xy = cell2pixel_coord(output[...,:2], scale)
  def loop_fn(carry, x):
    output, anchor = x
    return None, (output[...,2:3] * anchor[0], output[...,3:4] * anchor[1])
  _, (w, h) = jax.lax.scan(loop_fn, None, (output, anchors))
  return jnp.concatenate([xy, w, h, output[...,4:]], axis=-1)

import numpy as np
def test_target_show(image, target, mask, anchors):
  result_bboxes = []
  for i in range(3):
    # org_target = target[i]
    cvt_target = cell2pixel(target[i], scale=2**(i+3), anchors=anchors[i])
    for j in range(3):
      # print(f"Anchor {j} params in scale {2**(i+3)} (org target):", org_target[j][(org_target[j,...,4]==1) & (org_target[j,...,5+58]==1)])
      # print(f"Anchor {j} params in scale {2**(i+3)} (cvt target):", cvt_target[j][(cvt_target[j,...,4]==1) & (cvt_target[j,...,5+0]==1)])
      params = cvt_target[j][cvt_target[j,...,4]==1]  # (N,5+num_classes)
      for x,y,w,h,c,label in params:
        result_bboxes.append(np.stack([x,y,w,h,label]))
  if len(result_bboxes):
    result_bboxes = np.stack(result_bboxes)
  print("num bbox:", len(result_bboxes))
  print("mask postive num:", sum([(1-mask[i]).sum() for i in range(3)]))
  if args.path_dataset.name.lower() == 'coco':
    from katacv.utils.coco.build_dataset import show_bbox
  if args.path_dataset.name.lower() == 'pascal':
    from katacv.utils.pascal.build_dataset import show_bbox
  show_bbox(image, result_bboxes)

if __name__ == '__main__':
  from katacv.yolov4.parser import get_args_and_writer
  args = get_args_and_writer(no_writer=True)
  args.batch_size = 4
  if args.path_dataset.name.lower() == 'coco':
    from katacv.utils.coco.build_dataset import DatasetBuilder
  if args.path_dataset.name.lower() == 'pascal':
    from katacv.utils.pascal.build_dataset import DatasetBuilder
  ds_builder = DatasetBuilder(args)
  ds = ds_builder.get_dataset(subset='train')
  bar = tqdm(ds)
  max_relative_w, max_relative_h = 0, 0
  ws, hs = [], []
  for images, params, num_bboxes in bar:
    images, params, num_bboxes = images.numpy(), params.numpy(), num_bboxes.numpy()
    pred = [jnp.empty((
        args.batch_size,
        3, args.image_shape[0]//(2**i),
        args.image_shape[0]//(2**i),
        5 + args.num_classes
      )) for i in range(3, 6)
    ]
    # print(params.shape, num_bboxes.shape, pred[0].shape)
    # print("total box num:", num_bboxes[0])
    # print(params[0][:num_bboxes[0]])
    # target, mask = build_target(params[0], num_bboxes[0], [x[0] for x in pred], args.anchors)
    target, mask = jax.vmap(build_target, in_axes=(0,0,0,None), out_axes=0)(
      params, num_bboxes, pred, args.anchors
    )
    for i in range(3):
      obj_target = target[i][target[i][...,4]==1]
      ws.append(obj_target[:,2])
      hs.append(obj_target[:,3])
      max_relative_w = max(max_relative_w, jnp.max(target[i][..., 2]))
      max_relative_h = max(max_relative_h, jnp.max(target[i][..., 3]))
      # print(target[i].shape, mask[i].shape)
    bar.set_description(f"max w: {max_relative_w:.2f}, max h: {max_relative_h:.2f}")

    # Target show
    for i in range(args.batch_size):
      test_target_show(images[i], [x[i] for x in target], mask, args.anchors)
    break

  # with open("./logs/target_wh.npy", 'wb') as file:
  #   ws = np.concatenate(ws, axis=0)
  #   hs = np.concatenate(hs, axis=0)
  #   np.save(file, {'ws': ws, 'hs': hs}, allow_pickle=True)
  # print(max_relative_w, max_relative_h)
