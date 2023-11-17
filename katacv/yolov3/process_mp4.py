import sys, os
sys.path.append(os.getcwd())

from katacv.utils.related_pkgs.jax_flax_optax_orbax import *
from katacv.utils.related_pkgs.utility import *

from PIL import Image
import numpy as np

def main(args):
    from katacv.yolov3.parser import get_args_and_writer
    from katacv.yolov3.yolov3_model import get_yolov3_state
    yolov3_args = get_args_and_writer(no_writer=True, input_args="")
    yolov3_args.C = args.num_class
    print("Get model state and load weights...")
    state = get_yolov3_state(yolov3_args)
    weights = ocp.PyTreeCheckpointer().restore(str(args.path_weights))
    state = state.replace(params=weights['params'], params_darknet=weights['params_darknet'], batch_stats=weights['batch_stats'])
    print("Load state complete!")

    import moviepy.editor as mp
    input_video = str(args.path_input_video)
    output_video = str(args.path_output_video)

    clip = mp.VideoFileClip(input_video)
    origin_fps = clip.fps
    origin_duration = clip.duration

    processed_frames = []

    from katacv.utils.detection import plot_box_PIL, nms_boxes_and_mask
    from katacv.utils.VOC.label2realname import label2realname
    from katacv.yolov3.yolov3_predict import predict
    label2realname = label2realname[args.label_type]

    @jax.jit
    def predict_and_nms(x):
        boxes = predict(state, x, yolov3_args.anchors)[0]
        boxes, mask = nms_boxes_and_mask(boxes, iou_threshold=0.45, conf_threshold=0.2)
        return boxes, mask

    print("Compile XLA...")
    boxes, mask = predict_and_nms(jnp.zeros((1,416,416,3), dtype='float32'))
    print("Compile complete!")

    bar = tqdm(clip.iter_frames(), total=math.ceil(origin_fps * origin_duration))
    SPS_avg, fps_avg = 0, 0
    for idx, frame in enumerate(bar):
        start_time = time.time()
        x = jax.image.resize(frame[None,...], (1,416,416,3), method="bilinear")
        boxes, mask = predict_and_nms(x)
        SPS_avg += (1/(time.time() - start_time) - SPS_avg) / (idx+1)
        boxes = np.array(boxes[mask])
        image = Image.fromarray(frame)
        for i in range(boxes.shape[0]):
            image = plot_box_PIL(image, boxes[i,1:5], text=f"{label2realname[int(boxes[i,5])]} {float(boxes[i,0]):.2f}")
        processed_frames.append(np.array(image))
        fps_avg += (1/(time.time() - start_time) - fps_avg) / (idx+1)
        bar.set_description(f"SPS:{SPS_avg:.2f} fps:{fps_avg:.2f}")
        # if idx % 30 == 0:
        #     image.show()

    processed_clip = mp.ImageSequenceClip(processed_frames, fps=30)
    processed_clip.write_videofile(output_video)

def parse_args():
    from katacv.utils.parser import cvt2Path
    parser = argparse.ArgumentParser()
    parser.add_argument("--label-type", type=str, default="PASCAL",
        help="The dataset name in `katacv/utils/VOC/label2realname`.")
    parser.add_argument("--num-class", type=int, default=20,
        help="The number of classes in model.")
    # parser.add_argument("--path-weights", type=cvt2Path, default=Path("/home/wty/Coding/models/YOLOv3/YOLOv3-PASCAL-0080-lite"),
    parser.add_argument("--path-weights", type=cvt2Path, default=Path("/home/yy/Coding/models/YOLOv3/YOLOv3-PASCAL-0080-lite"),
        help="The path of model weights.")
    # parser.add_argument("--path-input-video", type=cvt2Path, default=Path("/home/wty/Videos/model_test/2.mp4"),
    parser.add_argument("--path-input-video", type=cvt2Path, default=Path("/home/yy/Videos/model_test/2.mp4"),
        help="The path of the input video.")
    parser.add_argument("--path-output-video", type=cvt2Path, default=None,
        help="The path of the output video, default 'path_input_video+fname_yolov3.mp4'")
    args = parser.parse_args()
    if args.path_output_video is None:
        fname = args.path_input_video.name
        suffix = fname.split('.')[-1]
        fname = fname[:-len(suffix)-1]
        args.path_output_video = args.path_input_video.parent.joinpath(fname + "_yolov3" + '.' + suffix)
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)
