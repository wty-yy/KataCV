from pathlib import Path
import sys
sys.path.append(str(Path.cwd().parent.parent))

from katacv.utils.related_pkgs.jax_flax_optax_orbax import *
from katacv.utils.imagenet.train import TrainState
from katacv.yolov4.yolov4 import logits2cell
from katacv.yolov4.build_yolo_target import cell2pixel
from katacv.yolov4.metric import logits2prob_from_list, get_pred_bboxes, show_bbox, calc_AP50_AP75_AP, mAP, coco_mAP

@jax.jit
def predict(state: TrainState, images: jax.Array):
  logits = state.apply_fn(
    {'params': state.params, 'batch_stats': state.batch_stats},
    images, train=False
  )
  pred_cell = [logits2cell(logits[i]) for i in range(3)]
  pred_pixel = [jax.vmap(cell2pixel, in_axes=(0,None,None), out_axes=0)(
    pred_cell[i], 2**(i+3), args.anchors[i]
  ) for i in range(3)
  ]
  pred_pixel_prob = logits2prob_from_list(pred_pixel)
  return pred_pixel_prob

if __name__ == '__main__':
  from katacv.yolov4.parser import get_args_and_writer
  # args = get_args_and_writer(no_writer=True, input_args="--model-name YOLOv4-mse --load-id 82".split())
  args = get_args_and_writer(no_writer=True, input_args="--model-name YOLOv4 --load-id 16".split())
  args.batch_size = 1
  args.path_cp = Path("/home/wty/Coding/GitHub/KataCV/logs/YOLOv4-checkpoints")
  # args.path_cp = Path("/home/wty/Coding/GitHub/KataCV/logs/YOLOv4-mse-checkpoints")

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

  val_iter = iter(val_ds)
  # sample_iter = iter(sample_ds)
  # images, bboxes, num_bboxes = next(sample_iter)
  # images, bboxes, num_bboxes = next(val_iter)

  test_num = 10

  # for images, bboxes, num_bboxes in val_ds:
  for images, bboxes, num_bboxes in train_ds:
    images, bboxes, num_bboxes = images.numpy(), bboxes.numpy(), num_bboxes.numpy()
    pred = predict(state, images)

    # print(jnp.sort(pred[0,:,4])[::-1][:50])
    # print(bboxes[0][:num_bboxes[0]])
    # from katacv.utils.coco.build_dataset import show_bbox
    import numpy as np
    np.set_printoptions(suppress=True)
    pred_bboxes = get_pred_bboxes(pred, conf_threshold=0.05, iou_threshold=0.4)
    for i in range(len(pred_bboxes)):
      # print(np.round(np.array(pred_bboxes[i]), 4))
      # print("Predict box num:", len(pred_bboxes[i]))
      show_bbox(images[i], pred_bboxes[i], args.path_dataset.name)
      AP50 = mAP(pred_bboxes[i], bboxes[i][:num_bboxes[i]], iou_threshold=0.5)
      AP75 = mAP(pred_bboxes[i], bboxes[i][:num_bboxes[i]], iou_threshold=0.75)
      AP = coco_mAP(pred_bboxes[i], bboxes[i][:num_bboxes[i]])
      print(f"AP50: {AP50:.2f}, AP75: {AP75:.2f}, AP: {AP:.2f}")
      # break
    test_num -= 1
    if test_num == 0:
      break

    # print(calc_AP50_AP75_AP(pred_bboxes, bboxes, num_bboxes))

