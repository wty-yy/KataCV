# -*- coding: utf-8 -*-
'''
@File    : yolov4.py
@Time    : 2023/11/17 19:54:22
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    :
2023/11/17: YOLOv4 files:
- `config.py`: The config constants used in yolov4.
- `csp_darknet53.py`: The pretrain CSP-Darknet53 model in Imagenet2012,
  with top-1: 76.55%, top-5: 93.16%, 26 millons parameters (105.5MB).
- `yolov4_model.py`: The model of YOLOv4 (Neck and Head).
- `build_yolo_target.py`: Make the YOLO target from the data: (images, bboxes and labels).
- `logs.py`: The logs manager.

2023/11/18:
BUGs fix:
- mask_obj = target[...,4:5] == 0.0  # WTF?
- when calculate the loss, use mask at last, don't use to the input!
'''

import sys, os
sys.path.append(os.getcwd())
from katacv.utils.related_pkgs.utility import *
from katacv.utils.related_pkgs.jax_flax_optax_orbax import *

from katacv.utils.imagenet.train import TrainState
from katacv.utils.detection import iou
from katacv.yolov4.build_yolo_target import build_target, cell2pixel

def logits2cell(logits: jax.Array):
  # xy = (jax.nn.sigmoid(logits[...,:2]) - 0.5) * 1.1 + 0.5
  # wh = (jax.nn.sigmoid(logits[...,2:4])*2)**2
  xy = jax.nn.sigmoid(logits[...,:2])
  wh = jnp.exp(logits[...,2:4])
  return jnp.concatenate([xy, wh, logits[...,4:]], axis=-1)

@partial(jax.jit, static_argnames=['train'])
def model_step(
    state: TrainState,
    images: jax.Array,  # (N, H, W, C)
    bboxes: jax.Array,  # (N, M, 5): (x,y,w,h,label)
    num_bboxes: jax.Array,  # (N,)
    train: bool
):
  def single_loss_fn(logits, target, mask_noobj, anchors):
    """
    pred.shape = (N, 3, H, W, 5 + num_classes)
    target.shape = (N, 3, H, W, 6)
    mask_noobj.shape = (N, 3, H, W, 1)
    anchors.shape = (3, 2)
    """
    def bce(logits, y, mask):  # binary cross-entropy
      return -(mask * (
        y*jax.nn.log_sigmoid(logits)
        + (1-y)*jax.nn.log_sigmoid(-logits)
      )).sum((1,2,3,4)).mean()

    def ciou(bbox1, bbox2, mask):  # Complete IOU loss
      ciou_fn = partial(iou, format='ciou', keepdim=True)
      return (mask * (1 - jax.vmap(
        ciou_fn, in_axes=(0,0), out_axes=0
      )(bbox1, bbox2))).sum((1,2,3,4)).mean()
    
    def mse(logits, y, mask):
      return (mask * 0.5 * (logits - y) ** 2).sum((1,2,3,4)).mean()

    def ce(logits, y_sparse, mask):  # cross-entropy
      y_onehot = jax.nn.one_hot(y_sparse, num_classes=logits.shape[-1])
      return -(mask * (jax.nn.log_softmax(logits) * y_onehot)).sum((1,2,3,4)).mean()

    mask_obj = target[...,4:5] == 1.0

    ### no object loss ###
    # print("1:", mask_noobj.shape, mask_obj.shape, pred_cell.shape)
    loss_noobj = bce(logits[...,4:5], 0.0, mask_noobj)
    ### coordinate loss ###
    ### CIOU Loss -------------------------------------------------------------------
    # ws, hs = [], []  # BUG FIX: calculate the CIOU, wh need multiply anchors first
    # for i in range(3):
    #   ws.append(pred_cell[:,i:i+1,:,:,2:3] * anchors[i,0])
    #   hs.append(pred_cell[:,i:i+1,:,:,3:4] * anchors[i,1])
    # w = jnp.concatenate(ws, axis=1)
    # h = jnp.concatenate(hs, axis=1)
    # loss_coord = ciou(jnp.concatenate([pred_cell[..., :2], w, h], axis=-1), target[..., :4], mask_obj)
    ### MSE Loss -------------------------------------------------------------------
    loss_coord = (  # BUG FIX: use bce for xy, so don't use sigmoid first.
      bce(logits[...,:2], target[...,:2], mask_obj) + 
      mse(logits[...,2:4], target[...,2:4], mask_obj)
    )
    ### object loss ###
    loss_obj = bce(logits[...,4:5], 1.0, mask_obj)
    ### class loss ###
    loss_class = ce(logits[..., 5:], target[..., 5], mask_obj)

    loss = (
      args.coef_noobj * loss_noobj +
      args.coef_coord * loss_coord +
      args.coef_obj * loss_obj +
      args.coef_class * loss_class
    )
    return loss, (loss_noobj, loss_coord, loss_obj, loss_class)

  def loss_fn(params):
    logits, updates = state.apply_fn(
      {'params': params, 'batch_stats': state.batch_stats},
      images, train=train, mutable=['batch_stats']
    )
    pred_cell = [logits2cell(logits[i]) for i in range(3)]
    pred_pixel = jax.lax.stop_gradient([
      jax.vmap(cell2pixel, in_axes=(0,None,None), out_axes=0)(
        pred_cell[i], 2**(i+3), args.anchors[i]
      )
      for i in range(3)
    ])
    targets, mask_noobjs = jax.vmap(
      build_target, in_axes=(0,0,0,None), out_axes=0
    )(bboxes, num_bboxes, pred_pixel, args.anchors)
    # targets, mask_noobjs = jax.vmap(
    #   build_target, in_axes=(0,0,0,None), out_axes=0
    # )(bboxes, num_bboxes, pred_pixel, args.anchors)
    total_loss = 0
    losses = [0, 0, 0, 0]
    for i in range(3):
      loss, other_losses = single_loss_fn(logits[i], targets[i], mask_noobjs[i], args.anchors[i])
      total_loss += loss
      # total_loss += single_loss_fn(pred_cell[i], targets[i], mask_noobjs[i])
      for j in range(4):
        losses[j] = losses[j] + other_losses[j]
    l2_decay = 0.5 * sum(
      jnp.sum(x**2) for x in jax.tree_util.tree_leaves(params) if x.ndim > 1
    )
    total_loss = total_loss + args.weight_decay * l2_decay
    return total_loss, (updates, pred_pixel, losses)  # metrics
  
  if train:
    (loss, (updates, *metrics)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates['batch_stats'])
  else:
    loss, (_, *metrics) = loss_fn(state.params)
  return state, (loss, *metrics)

if __name__ == '__main__':
  ### Initialize arguments and tensorboard writer ###
  from katacv.yolov4.parser import get_args_and_writer
  args, writer = get_args_and_writer()
  
  ### Initialize log manager ###
  from katacv.yolov4.logs import logs

  ### Initialize model state ###
  from katacv.yolov4.yolov4_model import get_yolov4_state
  state = get_yolov4_state(args)

  ### Load weights ###
  from katacv.utils.model_weights import load_weights
  if args.load_id > 0:
    state = load_weights(state, args)
  else:
    darknet_weights = ocp.PyTreeCheckpointer().restore(str(args.path_darknet_weights))
    state.params['CSPDarkNet_0'] = darknet_weights['params']['darknet']
    state.batch_stats['CSPDarkNet_0'] = darknet_weights['batch_stats']['darknet']
    print(f"Successfully load CSP-DarkNet53 from '{str(args.path_darknet_weights)}'")

  ### Save config ###
  from katacv.utils.model_weights import SaveWeightsManager
  save_weight = SaveWeightsManager(args, ignore_exist=True, max_to_keep=2)
  
  if args.path_dataset.name.lower() == 'coco':
    from katacv.utils.coco.build_dataset import DatasetBuilder
  if args.path_dataset.name.lower() == 'pascal':
    from katacv.utils.pascal.build_dataset import DatasetBuilder
  ds_builder = DatasetBuilder(args)
  train_ds = ds_builder.get_dataset(subset='train')
  # train_ds = ds_builder.get_dataset(subset='sample')
  val_ds = ds_builder.get_dataset(subset='val')
  # val_ds = ds_builder.get_dataset(subset='sample')

  ### Train and evaluate ###
  # from katacv.yolov4.metric import logits2prob_from_list, get_pred_bboxes, calc_AP50_AP75_AP
  start_time, global_step = time.time(), 0
  if args.train:
    for epoch in range(state.step//len(train_ds)+1, args.total_epochs+1):
      print(f"epoch: {epoch}/{args.total_epochs}")
      print("training...")
      logs.reset()
      bar = tqdm(train_ds)
      # num_objs = []
      for images, bboxes, num_bboxes in bar:
        images, bboxes, num_bboxes = images.numpy(), bboxes.numpy(), num_bboxes.numpy()
        global_step += 1
        state, (loss, pred_pixel, other_losses) = model_step(state, images, bboxes, num_bboxes, train=True)
        # num_objs.append(int(num_obj))
        logs.update(
          ['loss_train', 'loss_noobj_train', 'loss_coord_train', 'loss_obj_train', 'loss_class_train'],
          [loss, *other_losses]
        )
        bar.set_description(f"loss={loss:.4f}, lr={args.learning_rate_fn(state.step):.8f}")
        if global_step % args.write_tensorboard_freq == 0:
          logs.update(
            ['SPS', 'SPS_avg', 'epoch', 'learning_rate'],
            [
              args.write_tensorboard_freq/logs.get_time_length(),
              global_step/(time.time()-start_time),
              epoch,
              args.learning_rate_fn(state.step),
            ]
          )
          logs.writer_tensorboard(writer, global_step)
          logs.reset()
      print("validating...")
      logs.reset()
      for images, bboxes, num_bboxes in tqdm(val_ds):
        images, bboxes, num_bboxes = images.numpy(), bboxes.numpy(), num_bboxes.numpy()
        global_step += 1
        _, (loss, pred_pixel, other_losses) = model_step(state, images, bboxes, num_bboxes, train=False)
        # pred = logits2prob_from_list(pred_pixel)  # (N, -1, 6)
        # pred_bboxes = get_pred_bboxes(pred)
        # ap50, ap75, ap = calc_AP50_AP75_AP(pred_bboxes, bboxes, num_bboxes)
        logs.update(
          [
            'loss_val', 'loss_noobj_val', 'loss_coord_val', 'loss_obj_val', 'loss_class_val',
            #  'AP50_val', 'AP75_val', 'AP_val',
            'epoch', 'learning_rate'
          ],
          [
            loss, *other_losses,
            #  ap50, ap75, ap,
            epoch, args.learning_rate_fn(state.step)
          ]
        )
      logs.writer_tensorboard(writer, global_step)
      
      ### Save weights ###
      if epoch % args.save_weights_freq == 0:
        save_weight(state)
  writer.close()
