import sys, os
sys.path.append(os.getcwd())

from katacv.utils.related_pkgs.jax_flax_optax_orbax import *
from katacv.utils.related_pkgs.utility import *
from katacv.yolov4.predict import predict, show_bbox
from katacv.yolov4.metric import get_pred_bboxes

from PIL import Image
import numpy as np

def load_model_state():
  from katacv.yolov4.parser import get_args_and_writer
  state_args = get_args_and_writer(no_writer=True, input_args="--model-name YOLOv4 --load-id 218".split())
  state_args.batch_size = 1
  state_args.path_cp = Path("/home/yy/Coding/GitHub/KataCV/logs/YOLOv4-checkpoints")

  from katacv.yolov4.yolov4_model import get_yolov4_state
  state_args.steps_per_epoch = 10  # any number
  state = get_yolov4_state(state_args)

  from katacv.utils.model_weights import load_weights
  state = load_weights(state, state_args)
  return state, state_args

def main(args):
  state, state_args = load_model_state()

  import moviepy.editor as mp
  input_video = str(args.path_input_video)
  output_video = str(args.path_output_video)

  clip = mp.VideoFileClip(input_video)
  image_size = clip.size
  origin_fps = clip.fps
  origin_duration = clip.duration

  processed_frames = []

  print("Compile XLA...")
  from katacv.yolov4.metric import nms_boxes_and_mask
  @jax.jit
  def preprocess(image):
    x = jnp.array(image)[None, ...]
    x = (x / np.max(x, axis=(0,1), keepdims=True)).astype(np.float32)
    x = jax.image.resize(x, (1,*state_args.image_shape), method='bilinear')
    pred = predict(state, x, state_args.anchors)[0]
    pred_bboxes, mask = nms_boxes_and_mask(
      pred, iou_threshold=0.4, conf_threshold=0.2, iou_format='diou'
    )
    return pred_bboxes, mask
  def predict_and_nms(image: jax.Array):
    pred_bboxes, mask = jax.device_get(preprocess(image))
    pred_bboxes = pred_bboxes[mask]
    pred_bboxes[:,0] = pred_bboxes[:,0] / state_args.image_shape[1] * image.shape[1]
    pred_bboxes[:,2] = pred_bboxes[:,2] / state_args.image_shape[1] * image.shape[1]
    pred_bboxes[:,1] = pred_bboxes[:,1] / state_args.image_shape[0] * image.shape[0]
    pred_bboxes[:,3] = pred_bboxes[:,3] / state_args.image_shape[0] * image.shape[0]
    return pred_bboxes
  predict_and_nms(jnp.zeros((state_args.image_shape), dtype='float32'))
  print("Compile complete!")

  bar = tqdm(clip.iter_frames(), total=math.ceil(origin_fps * origin_duration))
  SPS_avg, fps_avg = 0, 0
  for idx, frame in enumerate(bar):
    w_mid = int(frame.shape[1]/2.0)  # width center clip
    x1 = frame[:,:w_mid,:]
    x2 = frame[:,w_mid:,:]

    start_time = time.time()
    # Width predict
    bboxes1 = predict_and_nms(x1)
    bboxes2 = predict_and_nms(x2)
    bboxes2[:,0] += w_mid
    bboxes = np.concatenate([bboxes1,bboxes2], axis=0)
    # bboxes = predict_and_nms(frame)
    SPS_avg += (1/(time.time() - start_time) - SPS_avg) / (idx+1)

    image = show_bbox(frame/255.0, bboxes, show_image=False, dataset='coco')
    processed_frames.append(np.array(image))
    fps_avg += (1/(time.time() - start_time) - fps_avg) / (idx+1)
    bar.set_description(f"SPS:{SPS_avg:.2f} fps:{fps_avg:.2f}")
    # image.show()
    # break

  processed_clip = mp.ImageSequenceClip(processed_frames, fps=30)
  processed_clip.write_videofile(output_video)

def parse_args():
  from katacv.utils.parser import cvt2Path
  parser = argparse.ArgumentParser()
  # parser.add_argument("--path-input-video", type=cvt2Path, default=Path("/home/yy/Coding/datasets/CR/videos/fast_pig_2.6/OYASSU_20210528_episodes/1.mp4"),
  parser.add_argument("--path-input-video", type=cvt2Path, default=Path("/home/yy/Videos/model_test/4.mp4"),
    help="The path of the input video.")
  parser.add_argument("--path-output-video", type=cvt2Path, default=None,
    help="The path of the output video, default 'logs/processed_videos/fname_yolo.mp4'")
  args = parser.parse_args()
  if args.path_output_video is None:
    fname = args.path_input_video.name
    suffix = fname.split('.')[-1]
    fname = fname[:-len(suffix)-1]
    args.path_output_video = args.path_input_video.parent.joinpath(fname + "_yolo4" + '.' + suffix)
    # args.path_processed_videos = Path.cwd().joinpath("logs/processed_videos")
    # args.path_processed_videos.mkdir(exist_ok=True)
    # args.path_output_video = args.path_processed_videos.joinpath(fname + "_yolo4" + '.' + suffix)
  return args

if __name__ == '__main__':
  args = parse_args()
  main(args)
