from typing import Callable
import jax, jax.numpy as jnp
import flax, flax.linen as nn
from yolov1 import get_yolov1_state
from katacv.utils.detection import nms, get_best_boxes_and_classes, plot_box, plot_cells
from katacv.utils.VOC.label2realname import label2realname
from PIL import Image
from katacv.utils.parser import Parser
import matplotlib.pyplot as plt
import numpy as np
import time

def parse_args():
    parser = Parser()
    parser.add_argument("--split-size", type=int, default=7,
        help="the split size of the cells")
    parser.add_argument("--class-num", type=int, default=20,
        help="the number of the classes (labels)")
    parser.add_argument("--bounding-box", type=int, default=2,
        help="the number of bounding box in each cell")
    args = parser.parse_args()
    args.S, args.B, args.C = args.split_size, args.bounding_box, args.class_num
    args.load_id = 40
    args.input_shape = (1, 448, 448, 3)
    args.path_images = args.path_logs.joinpath("test_images")
    args.path_load_weights = args.path_logs.joinpath(f"YoloV1-checkpoints/YoloV1-{args.load_id:04}")
    return args

@jax.jit
def predict(state, x):
    proba, boxes = state.apply_fn(
        {'params': state.params, 'batch_stats': state.batch_stats},
        x, train=False,
    )
    return jax.device_get(proba), jax.device_get(boxes)

if __name__ == '__main__':
    args = parse_args()
    state = get_yolov1_state(args)
    with open(args.path_load_weights, 'rb') as file:
        state = flax.serialization.from_bytes(state, file.read())
    print("Start predict...")
    while True:
        time.sleep(0.5)
        for file in args.path_images.iterdir():
            if not file.is_file(): continue
            print(f"Processing image '{str(file)}'...")
            origin_image = Image.open(str(file))
            resize_image = origin_image.resize((448, 448))
            origin_image = np.array(origin_image)
            x = np.expand_dims(np.array(resize_image), 0)
            proba, boxes = predict(state, x)
            cells = jnp.concatenate([proba, boxes], axis=-1)
            # print(boxes)
            boxes = get_best_boxes_and_classes(cells, args.S, args.B, args.C)[0]  # (SxS)x6

            # boxes = boxes[boxes[:,5] == 14]
            # print(boxes)
            fig, ax = plt.subplots(1, 2, figsize=(10,6))
            ax[0].imshow(origin_image)
            for i in range(boxes.shape[0]):
                plot_box(ax[0], origin_image.shape, boxes[i,1:5], text=f"{label2realname[int(boxes[i,5])]} {float(boxes[i,0]):.2f}")
            # plt.show()

            boxes = nms(boxes, iou_threshold=0.3, conf_threshold=0.1)

            # fig, ax = plt.subplots(figsize=(20,20))
            ax[1].imshow(origin_image)
            for i in range(boxes.shape[0]):
                plot_box(ax[1], origin_image.shape, boxes[i,1:5], text=f"{label2realname[int(boxes[i,5])]} {float(boxes[i,0]):.2f}")
            plt.show()

            # plot_cells(ax, origin_image.shape, args.S)
            

