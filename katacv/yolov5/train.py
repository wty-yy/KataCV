# -*- coding: utf-8 -*-
'''
@File    : train.py
@Time    : 2023/12/13 11:22:37
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : 
2023/12/22: Start training on my RTX4080.
Fix bugs:
1. Calculating mean loss should use mask.sum() as denominator.
2. Forget stopping gradient when calculating object loss.
3. Fix the prediction function for confidence calculating.
4. Update self.state in predictor for evaluating metrics in-time [must pass new state].
2023/12/23: Use 30 batch size, 97% GPU memory
2023/12/25: Training 79 epochs found no weight decay and gradient norm clip (max_norm=10.0)!
2023/12/26: Update nms iou_thre=0.65, use IOU metrics (old: DIOU),
  add more buffer `max_num_box*30` to nms (old: `max_num_box*9`)
2023/12/27: Update CIOU: `wh` relative to cell.
2023/12/29: Found mAP, AP50, AP75 jump huge after 40 epochs.
  1. Add stopping gradient of DIOU diagonal distance.
  2. Add accumulate gradient to nominal batch size 64. (start train 16 batch size)
2023/12/30: FIX BUG:
  1. Fix weight decay coef size for accumulating gradient.
2024/1/1: FIX BUG:
  1. Loss calculate `train=train`. (Huge different result whether turn on the batch normalize)
  2. Check paper https://arxiv.org/pdf/1906.07155.pdf section 5.2,\ 
    when use pretrain backbone model, we must freeze BN statistic in backbone model,\ 
    also we can use 2x learning rate.
2024/1/9: Complete training from scratch (on RTX 4090): batch=32, nominal batch=64, val result:
p: 0.488 r: 0.612 ap50: 0.559 ap75: 0.402 map: 0.379: 100%|██████████| 156/156 [01:28<00:00,  1.76it/s]
2024/1/15: Fix BUG:
1. backbone stage size = [3,6,9,3] and CSP bottleneck channel is output_channel // 2
  YOLOv5: Total Parameters: 46,623,741 (186.5 MB)
Add:
1. Two different learning rate schedule for 'bias' and other weights.
Modify:
1. Change decayed weights to optax, should be faster.
'''
import sys, os
sys.path.append(os.getcwd())
from katacv.utils.related_pkgs.utility import *
from katacv.utils.related_pkgs.jax_flax_optax_orbax import *
import numpy as np

if __name__ == '__main__':
  ### Initialize arguments and tensorboard writer ###
  from katacv.yolov5.parser import get_args_and_writer
  args, writer = get_args_and_writer()
  
  ### Initialize log manager ###
  from katacv.yolov5.logs import logs

  ### Initialize model state ###
  from katacv.yolov5.model import get_state
  state = get_state(args, use_init=not args.load_id)

  ### Load weights ###
  from katacv.utils.model_weights import load_weights
  if args.load_id > 0:
    state = load_weights(state, args)
  elif args.pretrain_backbone:
    darknet_weights = ocp.PyTreeCheckpointer().restore(str(args.path_darknet_weights))
    state.params['CSPDarkNet_0'] = darknet_weights['params']['darknet']
    state.batch_stats['CSPDarkNet_0'] = darknet_weights['batch_stats']['darknet']
    print(f"Successfully load CSP-DarkNet53 from '{str(args.path_darknet_weights)}'")
  else:
    print("Don't use pretrained backbone darknet weight, start from scratch.")

  ### Save config ###
  from katacv.utils.model_weights import SaveWeightsManager
  save_weight = SaveWeightsManager(args, ignore_exist=True, max_to_keep=2)
  
  from katacv.utils.yolo.build_dataset import DatasetBuilder
  ds_builder = DatasetBuilder(args)
  train_ds = ds_builder.get_dataset(subset='train', use_cache=False)
  val_ds = ds_builder.get_dataset(subset='val', use_cache=False)
  args.max_num_box = train_ds.dataset.max_num_box

  ### Build predictor for validation ###
  from katacv.yolov5.predict import Predictor
  predictor = Predictor(args, state)

  ### Build loss updater for training ###
  from katacv.yolov5.loss import ComputeLoss
  compute_loss = ComputeLoss(args)

  ### Train and evaluate ###
  start_time, global_step = time.time(), 0
  if args.train:
    # for epoch in range(state.step//len(train_ds)+1, args.total_epochs+1):
    for epoch in range(args.load_id+1, args.total_epochs+1):
      print(f"epoch: {epoch}/{args.total_epochs}")
      print("training...")
      logs.reset()
      bar = tqdm(train_ds)
      for x, tbox, tnum in bar:  # Normalize image x !
        x, tbox, tnum = x.numpy().astype(np.float32) / 255.0, tbox.numpy(), tnum.numpy()
        global_step += 1
        state, metrics = compute_loss.step(state, x, tbox, tnum, train=True)
        logs.update(
          [
            'loss_train', 'loss_box_train', 'loss_obj_train', 'loss_cls_train',
          ],
          metrics
        )
        bar.set_description(f"loss={metrics[0]:.4f}, lr={args.learning_rate_fn(state.step):.8f}")
        if global_step % args.write_tensorboard_freq == 0:
          logs.update(
            ['SPS', 'SPS_avg', 'epoch', 'learning_rate', 'learning_rate_bias'],
            [
              args.write_tensorboard_freq/logs.get_time_length(),
              global_step/(time.time()-start_time),
              epoch,
              args.learning_rate_fn(state.step),
              args.learning_rate_bias_fn(state.step)
            ]
          )
          logs.writer_tensorboard(writer, global_step)
          logs.reset()
      print("validating...")
      logs.reset()
      predictor.reset(state=state)
      for x, tbox, tnum in tqdm(val_ds):
        x, tbox, tnum = x.numpy().astype(np.float32) / 255.0, tbox.numpy(), tnum.numpy()
        predictor.update(x, tbox, tnum)
        _, metrics = compute_loss.step(state, x, tbox, tnum, train=False)
        logs.update(
          ['loss_val', 'loss_box_val', 'loss_obj_val', 'loss_cls_val'],
          metrics
        )
      p50, r50, ap50, ap75, map = predictor.p_r_ap50_ap75_map()
      for name, val in zip(['P@50_val', 'R@50_val', 'AP@50_val', 'AP@75_val', 'mAP_val'], [p50, r50, ap50, ap75, map]):
        print(f"{name}={val:.4f}", end=' ')
      print()
      logs.update(
        [
          'P@50_val', 'R@50_val', 'AP@50_val', 'AP@75_val', 'mAP_val',
          'epoch', 'learning_rate', 'learning_rate_bias'
        ],
        [
          p50, r50, ap50, ap75, map,
          epoch, args.learning_rate_fn(state.step),
          args.learning_rate_bias_fn(state.step)
        ]
      )
      logs.writer_tensorboard(writer, global_step)
      predictor.reset()
      
      ### Save weights ###
      if epoch % args.save_weights_freq == 0:
        save_weight(state)
  writer.close()
