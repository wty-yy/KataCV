# -*- coding: utf-8 -*-
'''
@File    : build_dataset.py
@Time    : 2023/09/24 12:21:29
@Author  : wty-yy
@Version : 1.0
@Blog    : https://wty-yy.space/
@Desc    : 
已处理好dCOCO数据集2014：https://www.kaggle.com/datasets/79abcc2659dc745fddfba1864438afb2fac3fabaa5f37daa8a51e36466db101e
2023/09/24: 完成数据集构造，需要使用到以下8个超参数
    path_dataset_tfrecord: Path
    batch_size: int
    shuffle_size: int
    image_size: int
    split_sizes: List[int]
    anchors: List[Tuple[int]]
    bounding_box: int
    iou_ignore_threshold: float
'''
from pathlib import Path
from typing import NamedTuple
import tensorflow as tf
from katacv.utils.related_pkgs.utility import *

def resize_and_augmentation(image: tf.Tensor, params: tf.Tensor, target_size: int, use_aug: bool = True):
    image = tf.image.resize([image], [target_size, target_size])[0]
    flip_rand = tf.cast(tf.random.uniform([]) < 0.5, tf.bool)
    if not use_aug: flip_rand = tf.constant(True)
    def flip_left_right():
        flip_image = tf.image.flip_left_right(image)
        flip_params = tf.concat([1.0 - params[:,0:1], params[:,1:]], axis=1)
        return flip_image, flip_params
    image, params = tf.cond(
        flip_rand,
        lambda: (image, params),
        flip_left_right
    )
    image = tf.cast(image, tf.uint8)
    return image, params

def iou_relative(box1: tf.Tensor, box2: tf.Tensor):
    inter = tf.reduce_prod(tf.minimum(box1, box2))
    union = tf.reduce_prod(box1) + tf.reduce_prod(box2) - inter
    return inter / (union + 1e-6)

# @tf.function
def make_target(
        labels: tf.Tensor,
        params: tf.Tensor,
        splits: List[int],
        anchors: List[Tuple[int, int]],
        A: int,
        iou_ignore_threshold: float,
    ):
    """
    Inputs:
        let `N=tf.shape(labels)[0]` be the bounding box numbers in current example.
        - labels: `shape=(N,)`, the class label for each box.
        - params: `shape=(N,4)`, the parameters for each box.
        - splits: The split sizes of the model outputs.
        - anchors: The anchors bounding box with width and height.
        - A: The anchor number in each split size.
        - iou_ignore_threshold: the threshold of iou, 
        if the iou between anchor box and target box (not the biggest iou one) is bigger than 
        `iou_ignore_threshold`, then it will be ignore (this example will not add in loss).
    """
    target = [tf.zeros((S,S,A,6), tf.float32) for S in splits]
    for o in range(tf.shape(params)[0]):
        # sort args by iou
        ious = tf.convert_to_tensor([iou_relative(params[o,2:4], tf.constant(anchor)) for anchor in anchors])
        sort_args = tf.argsort(ious)[::-1]
        # has_rep = [False] * tf.size(splits)
        has_rep = tf.zeros_like(splits, tf.bool)
        for now in sort_args:
            s_i, a_i = now // A, now % A  # the idx of split_sizes and anchor in the split size
            S = tf.cast(tf.convert_to_tensor(splits)[s_i], tf.float32)
            # j, i = tf.cast(params[o,0:2]*S, tf.int32)
            j = tf.cast(params[o,0]*S, tf.int32)
            i = tf.cast(params[o,1]*S, tf.int32)
            if has_rep[s_i]: continue
            x_cell = params[o,0]*S - tf.cast(j, tf.float32)
            y_cell = params[o,1]*S - tf.cast(i, tf.float32)
            w_cell = params[o,2]*S
            h_cell = params[o,3]*S
            cls = tf.cast(labels[o], tf.float32)
            features = tf.stack([1.0, x_cell, y_cell, w_cell, h_cell, cls])
            now_t = None
            if tf.equal(s_i, 0): now_t = target[0]
            elif tf.equal(s_i, 1): now_t = target[1]
            else: now_t = target[2]
            used = now_t[i,j,a_i,0]
            # used = tf.gather(target, s_i)[i,j,a_i,0]
            if tf.equal(used, 0.0) and not has_rep[s_i]:  # positive example
                has_rep = tf.tensor_scatter_nd_update(has_rep, [(s_i,)], [True])
                update1 = lambda t: tf.tensor_scatter_nd_update(t, [(i,j,a_i,k) for k in range(6)], features)
                if tf.equal(s_i, 0): target[0] = update1(target[0])
                elif tf.equal(s_i, 1): target[1] = update1(target[1])
                else: target[2] = update1(target[2])
            elif tf.equal(used, 0) and ious[i] > iou_ignore_threshold:  # ignore example
                update2 = lambda t: tf.tensor_scatter_nd_update(t, [(i,j,a_i,0)], [-1])
                if tf.equal(s_i, 0): target[0] = update2(target[0])
                elif tf.equal(s_i, 1): target[1] = update2(target[1])
                else: target[2] = update2(target[2])
                # now_t = tf.tensor_scatter_nd_update(now_t, [(i,j,a_i,0)], [-1])  # use -1 to ignore the box
        # j, i = tf.cast(params[o,0]*S, tf.int32), tf.cast(params[o,1]*S, tf.int32)
        # x_cell = params[o,0]*S - tf.cast(j, tf.float32)
        # y_cell = params[o,1]*S - tf.cast(i, tf.float32)
        # w_cell = params[o,2]*S
        # h_cell = params[o,3]*S
        # cls = tf.cast(labels[o], tf.float32)
        # features = tf.stack([1.0, x_cell, y_cell, w_cell, h_cell, cls])
        # label = tf.tensor_scatter_nd_update(label, [(i,j,k) for k in range(6)], features)
    target = [tf.reshape(t, (-1, 6)) for t in target]
    target = tf.concat(target, axis=0)
    return target

def decode_example(target_size, split_sizes, anchors, anchor_per, iou_ignore_threshold, use_aug):
    def thunk(example):
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'labels': tf.io.VarLenFeature(tf.int64),
            'params': tf.io.VarLenFeature(tf.float32)
        }
        example = tf.io.parse_single_example(example, feature_description)
        image = tf.io.decode_jpeg(example['image'], channels=3)
        labels = tf.sparse.to_dense(example['labels'], default_value=0)
        params = tf.reshape(tf.sparse.to_dense(example['params'], default_value=0.0), [-1, 4])
        image, params = resize_and_augmentation(image, params, target_size, use_aug)
        label = make_target(labels, params, split_sizes, anchors, anchor_per, iou_ignore_threshold)
        return image, label
    return thunk

DATA_SIZE = {
    'train': 117264,  # 15:22/iter only load
    'val': 4954,  # 33 s/iter only load, 142 s/iter with preprocess
    '8examples': 8,  # 33 sec pre iter
}
class COCOBuilder():
    path_dataset_tfrecord: Path  # `COCO-train.tfrecord`, `COCO-val.tfrecord`, ...
    batch_size: int
    shuffle_size: int
    image_size: int
    split_sizes: List[int]
    anchors: List[Tuple[int]]
    anchor_per: int
    iou_ignore_threshold: float
    
    def __init__(self, args: NamedTuple):
        self.path_dataset_tfrecord = args.path_dataset_tfrecord
        self.batch_size, self.shuffle_size = args.batch_size, args.shuffle_size
        self.image_size = args.image_size
        self.split_sizes = args.split_sizes
        self.anchors = args.anchors
        self.anchor_per = args.bounding_box  # anchor_per is same as bounding_box number
        self.iou_ignore_threshold = args.iou_ignore_threshold
    
    def get_dataset(self, subset='train', repeat=1, shuffle=True, use_aug=True):
        ds_tfrecord = tf.data.TFRecordDataset(str(self.path_dataset_tfrecord.joinpath(f"COCO-{subset}.tfrecord")))
        ds = ds_tfrecord.map(
            decode_example(
                self.image_size, self.split_sizes,
                self.anchors, self.anchor_per,
                self.iou_ignore_threshold,
                use_aug)
        ).repeat(repeat)
        if shuffle: ds = ds.shuffle(self.shuffle_size)
        ds = ds.batch(self.batch_size, drop_remainder=True)
        return ds, DATA_SIZE[subset] * repeat // self.batch_size

def get_targets(args, y):
    """
    Inputs:
        Reshape the y of dataset to the list, include `len(args.split_sizes)` element.
        - `args`: The arguments with `split_sizes` and `anchor_per` variable.
        - `y`: The label of the dataset's example.

    Return:
        Let `N` be the batch size.
        The output's shape is `(N,S,S,anchor_per,6) for S in split_sizes`
    """
    targets, last_idx = [], 0
    for S in args.split_sizes:
        now_idx = last_idx + S*S*args.B
        targets.append(y[:,last_idx:now_idx,:].numpy().reshape(-1,S,S,args.B,6))
        last_idx = now_idx
    return targets

if __name__ == '__main__':
    import argparse
    import katacv.yolov3.constant_coco as const
    parser = argparse.ArgumentParser()
    cvt2Path = lambda x: Path(x)
    parser.add_argument("--path-dataset-tfrecord", type=cvt2Path, default=const.path_dataset_tfrecord)
    parser.add_argument("--batch-size", type=int, default=const.batch_size)
    parser.add_argument("--shuffle-size", type=int, default=const.shuffle_size)
    parser.add_argument("--image-size", type=int, default=const.image_size)
    parser.add_argument("--split-sizes", nargs=3, default=const.split_sizes)
    parser.add_argument("--anchors", default=const.anchors)
    parser.add_argument("--bounding-box", type=int, default=const.bounding_box)
    parser.add_argument("--iou-ignore-threshold", type=float, default=const.iou_ignore_threshold)
    args = parser.parse_args()
    args.B = args.bounding_box

    ds_builder = COCOBuilder(args)
    # ds, ds_size = ds_builder.get_dataset("train")
    ds, ds_size = ds_builder.get_dataset("8examples", shuffle=False)
    # ds, ds_size = ds_builder.get_dataset("val", use_aug=False)
    print("Datasize:", ds_size)

    from katacv.utils.detection import plot_box, plot_cells
    import time, numpy as np
    start_time = time.time()
    import matplotlib.pyplot as plt
    plt.rcParams['font.family'] = ['serif']
    from label2realname import label2realname
    from tqdm import tqdm

    count = 0
    for image, y in tqdm(ds, total=ds_size):
        count += 1
        # continue
        image = image.numpy()[0]
        targets = get_targets(args, y)
        # TODO: 验证label是否正确 OK
        fig, axs = plt.subplots(1,4,figsize=(20,6))
        # axs = axs.reshape(-1)
        for _ in range(3): axs[_].imshow(image)
        for idx, S in enumerate(args.split_sizes):
            ax = axs[idx]
            ax.axis('off')
            plot_cells(ax, image.shape, S)
            target = targets[idx][0]  # (S,S,3,6), (c,x,y,w,h,cls)
            # print("non zeros:", np.sum(target[:,:,:,0]))
            good_iou_count = 0
            for i in range(S):
                for j in range(S):
                    for k in range(args.bounding_box):
                        if target[i,j,k,0] == 1:
                            cat = target[i,j,k,5]
                            features = target[i,j,k,1:5]
                            x, y, w, h = (
                                (features[0]+j)/S,
                                (features[1]+i)/S,
                                features[2]/S,
                                features[3]/S
                            )
                            text = f"{label2realname[cat]} ({i},{j})"
                            iou_with_anchor = iou_relative(tf.constant((w,h), tf.float32), tf.constant(args.anchors[idx*3+k], tf.float32))
                            if iou_with_anchor > 0.35:
                                good_iou_count += 1
                                plot_box(ax, image.shape, (x, y, w, h), text)
                                # plot_box(ax, image.shape, (x, y, *args.anchors[idx*3+k]), f"anchor: {args.anchors[idx*3+k]}", box_color='green')
                                # plot_box(ax, image.shape, (x, y, *args.anchors[idx*3+k]), f"iou: {iou_with_anchor:.2f}", box_color='#176B87')
                                plot_box(ax, image.shape, (x, y, *args.anchors[idx*3+k]), f"anchor", box_color='#F86F03')
                            # print(target[i,j,:,0])
                            # assert(np.sum(target[i,j,:,0]) == 1)
            ax.set_title(f"split size: {S}, anchor iou > 0.35 num: {good_iou_count}")
        ax = axs[3]
        ax.imshow(image)
        boxes = []
        for idx, S in enumerate(args.split_sizes):
            target = targets[idx][0]  # (S,S,3,6), (c,x,y,w,h,cls)
            for i in range(S):
                for j in range(S):
                    for k in range(args.bounding_box):
                        if target[i,j,k,0] == 1:
                            cat = target[i,j,k,5]
                            features = target[i,j,k,1:5]
                            x, y, w, h = (
                                (features[0]+j)/S,
                                (features[1]+i)/S,
                                features[2]/S,
                                features[3]/S
                            )
                            boxes.append(np.array((x, y, w, h)))
                            text = f"{label2realname[cat]}"
                            plot_box(ax, image.shape, (x, y, w, h), text)
        eps = 1e-5
        boxes = np.round(np.array(boxes)/eps)*eps
        ax.set_title(f"Origin bboxes, total box num: {len(np.unique(boxes, axis=0))}")
        ax.axis('off')
        plt.tight_layout()
        plt.savefig("logs/test_build_dataset.jpg", dpi=200)
        plt.show()
    print("total data size:", count)
    print("used time:", time.time() - start_time)
