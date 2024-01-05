import sys, os
sys.path.append(os.getcwd())

from katacv.utils.related_pkgs.jax_flax_optax_orbax import *
from katacv.utils.related_pkgs.utility import *
from katacv.yolov5.predict import Predictor
from katacv.utils.yolo.utils import show_box

from PIL import Image
import numpy as np

def load_model_state():
  from katacv.yolov5.parser import get_args_and_writer
  state_args = get_args_and_writer(no_writer=True, input_args="--model-name YOLOv5_b32_scratch_stopD --load-id 164 --batch-size 1".split())

  from katacv.yolov5.model import get_state
  state = get_state(state_args, use_init=False)

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
  print(f"{image_size=}")
  origin_fps = clip.fps
  origin_duration = clip.duration

  processed_frames = []

  predictor = Predictor(state_args, state)
  @jax.jit
  def preprocess(x):
    w = jnp.array([x.shape[1] / state_args.image_shape[1], x.shape[0] / state_args.image_shape[0]])
    w = jnp.r_[w, w, 1, 1].reshape(1,1,6)
    x = jnp.array(x)[None, ...] / 255.
    x = jax.image.resize(x, (1,*state_args.image_shape), method='bilinear')
    pbox, pnum = predictor.pred_and_nms(state, x, iou_threshold=0.6, conf_threshold=0.2, nms_multi=10)
    pbox = pbox * w
    return pbox[0], pnum[0]
  
  def predict(x):
    pbox, pnum = jax.device_get(preprocess(x))
    pbox = pbox[:pnum]
    return pbox.copy()

  print("Compile XLA...")
  predict(np.zeros((*image_size[::-1],3), dtype=np.uint8))
  print("Compile complete!")

  bar = tqdm(clip.iter_frames(), total=math.ceil(origin_fps * origin_duration))
  SPS_avg, fps_avg = 0, 0
  # min_wh = np.array([1e5, 1e5], np.float32)
  for idx, frame in enumerate(bar):
    # frame = np.array(Image.fromarray(frame).resize((720, 1280)))
    w_mid = int(frame.shape[1]/2.0)  # width center clip
    x1 = frame[:,:w_mid,:]
    x2 = frame[:,w_mid:,:]

    start_time = time.time()

    # pbox1 = predict(x1)
    # pbox2 = predict(x2)
    # pbox2[:,0] += w_mid
    # pbox = np.concatenate([pbox1,pbox2], axis=0)
    pbox =  predict(frame)
    SPS_avg += (1/(time.time() - start_time) - SPS_avg) / (idx+1)

    image = show_box(frame, pbox, verbose=False, video=True)
    processed_frames.append(np.array(image))
    fps_avg += (1/(time.time() - start_time) - fps_avg) / (idx+1)
    # min_wh = np.minimum(min_wh, pbox[:,[2,3]].min(0))
    # bar.set_description(f"SPS:{SPS_avg:.2f} fps:{fps_avg:.2f} min:{min_wh.round(2)}")
    bar.set_description(f"SPS:{SPS_avg:.2f} fps:{fps_avg:.2f}")
    # image.show()
    # break

  processed_clip = mp.ImageSequenceClip(processed_frames, fps=30)
  processed_clip.write_videofile(output_video)

def parse_args():
  from katacv.utils.parser import cvt2Path
  parser = argparse.ArgumentParser()
  parser.add_argument("--path-input-video", type=cvt2Path, default=Path("/home/yy/Videos/model_test/2.mp4"),
    help="The path of the input video.")
  parser.add_argument("--path-output-video", type=cvt2Path, default=None,
    help="The path of the output video, default 'logs/processed_videos/fname_yolo.mp4'")
  args = parser.parse_args()
  if args.path_output_video is None:
    fname = args.path_input_video.name
    suffix = fname.split('.')[-1]
    fname = fname[:-len(suffix)-1]
    args.path_output_video = args.path_input_video.parent.joinpath(fname + "_yolov5_164" + '.' + suffix)
  return args

if __name__ == '__main__':
  args = parse_args()
  main(args)
