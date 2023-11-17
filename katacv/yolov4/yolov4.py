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
'''

import sys, os
sys.path.append(os.getcwd())
from katacv.utils.related_pkgs.utility import *
from katacv.utils.related_pkgs.jax_flax_optax_orbax import *

from katacv.utils.imagenet.train import TrainState
from katacv.utils.detection import iou
from katacv.yolov4.build_yolo_target import build_target, cell2pixel

def logits2cell(logits: jax.Array):
  xy = (jax.nn.sigmoid(logits[...,:2]) - 0.5) * 1.1 + 0.5
  wh = (jax.nn.sigmoid(logits[...,2:4])*2)**2
  return jnp.concatenate([xy, wh, logits[...,4:]], axis=-1)

@partial(jax.jit, static_argnames=['train'])
def model_step(
    state: TrainState,
    images: jax.Array,  # (N, H, W, C)
    bboxes: jax.Array,  # (N, M, 5): (x,y,w,h,label)
    num_bboxes: jax.Array,  # (N,)
    train: bool
):
  def single_loss_fn(pred_cell, target, mask_noobj):
    """
    pred.shape = (N, 3, H, W, 5 + num_classes)
    target.shape = (N, 3, H, W, 6)
    mask_noobj.shape = (N, 3, H, W, 1)
    """
    def bce(logits, y):  # binary cross-entropy, shape=(N,M)
      return -(
        y*jax.nn.log_sigmoid(logits)
        + (1-y)*jax.nn.log_sigmoid(-logits)
      ).sum(-1).mean()

    def ciou(bbox1, bbox2):  # Complete IOU loss, shape=(N,M,4)
      ciou_fn = partial(iou, format='ciou')
      return jax.vmap(
        ciou_fn, in_axes=(0,0), out_axes=0
      )(bbox1, bbox2).sum(-1).mean()

    def ce(logits, y_sparse):  # cross-entropy, shape=(N,M), (N,)
      y_onehot = jax.nn.one_hot(y_sparse, num_classes=logits.shape[-1])
      return -(jax.nn.log_softmax(logits), y_onehot).sum(-1).mean()

    N = pred_cell.shape[0]
    mask_obj = target[...,4:5] == 0.0

    ### no object loss ###
    loss_noobj = bce((mask_noobj*pred_cell)[...,4].reshape(N,-1), 0.0)
    ### coordinate loss ###
    loss_coord = ciou(
      (mask_obj * pred_cell)[..., :4].reshape(N, -1, 4),
      target[..., :4].reshape(N, -1, 4)
    )
    ### object loss ###
    loss_obj = bce((mask_obj*pred_cell)[...,4].reshape(N,-1), 1.0)
    ### class loss ###
    loss_class = ce(
      (mask_obj * pred_cell)[..., 5:].reshape(N, -1),
      target[..., 5].reshape(N,-1)
    )

    loss = loss_noobj + loss_coord + loss_obj + loss_class
    return loss

  def loss_fn(params):
    logits, updates = state.apply_fn(
      {'params': params, 'batch_stats': state.batch_stats},
      images, train=train, mutable=['batch_stats']
    )
    pred_cell = [logits2cell(logits[i]) for i in range(3)]
    pred_pixel = jax.lax.stop_gradient([
      cell2pixel(pred_cell[i], 2**(i+3), args.anchors[i])
      for i in range(3)
    ])
    targets, mask_noobjs = jax.vmap(
      build_target, in_axes=(0,0,0,None), out_axes=0
    )(bboxes, num_bboxes, pred_pixel, args.anchors)
    total_loss = 0
    for i in range(3):
      total_loss += single_loss_fn(pred_cell[i], targets[i], mask_noobjs[i])
    l2_decay = 0.5 * sum(
      jnp.sum(x**2) for x in jax.tree_util.tree_leaves(params) if x.ndim > 1
    )
    total_loss = total_loss + args.weight_decay * l2_decay
    return total_loss, (updates, pred_pixel)
  
  if train:
    (loss, (updates, pred_pixel)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    state = state.apply_gradients(grads=grads)
    state = state.replace(batch_stats=updates['batch_stats'])
  else:
    loss, (_, pred_pixel) = loss_fn(state.params)
  return state, (loss, pred_pixel)

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
  from katacv.utils import load_weights
  if args.load_id > 0:
    state = load_weights(state, args)
  else:
    darknet_weights = ocp.PyTreeCheckpointer().restore(str(args.path_darknet_weights))
    state.params['CSPDarkNet_0'] = darknet_weights['params']['darknet']
    state.batch_stats['CSPDarkNet_0'] = darknet_weights['batch_stats']['darknet']
    print(f"Successfully load CSP-DarkNet53 from '{str(args.path_darknet_weights)}'")

  ### Save config ###
  from katacv.utils import SaveWeightsManager
  save_weight = SaveWeightsManager(args, ignore=True)
  
  from katacv.utils.coco.build_dataset import DatasetBuilder
  ds_builder = DatasetBuilder(args)
  train_ds = ds_builder.get_dataset(subset='train')
  val_ds = ds_builder.get_dataset(subset='val')

  ### Train and evaluate ###
  start_time, global_step = time.time(), 0
  if args.train:
    for epoch in range(state.step//len(train_ds)+1, args.total_epochs+1):
      print(f"epoch: {epoch}/{args.total_epochs}")
      print("training...")
      logs.reset()
      for x, y in tqdm(train_ds):
        x, y = x.numpy(), y.numpy()
        global_step += 1
        state, (loss, pred) = model_step(state, x, y, train=True)
        logs.update(
          ['loss_train'], loss
        )
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
      for x, y in tqdm(val_ds):
        x, y = x.numpy(), y.numpy()
        _, (loss, pred) = model_step(state, x, y, train=False)
        logs.update(
          ['loss_val', 'epoch', 'learning_rate'],
          [loss, epoch, args.learning_rate_fn(state.step)]
        )
      logs.writer_tensorboard(writer, global_step)
      
      ### Save weights ###
      if epoch % args.save_weights_freq == 0:
        save_weight(state)
  writer.close()
