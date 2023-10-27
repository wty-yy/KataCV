# KataCV

> 原replicate-papers改名为KataCV，用于复现经典CV文献，当前用于个人测试

## 图像分类

| 模型      | 数据集   | 完成情况                           | 框架 |      Paper/Note|
| --------- | -------- | ---------------------------------- | ---- |  ----|
| LeNet-5   | MNIST    | OK                                 | TF2  | [Paper](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)     |
| AlexNet   | ImageNet | OK 参数存于网盘                    | TF2  | [Paper](https://proceedings.neurips.cc/paper_files/paper/2012/file/c399862d3b9d6b76c8436e924a68c45b-Paper.pdf)       |
| VGG16     | ImageNet | OK（但是调参有问题，初始lr过大） | JAX  | [Paper](https://arxiv.org/pdf/1409.1556.pdf)  |
| GoogleNet | ImageNet | OK  (val-top5: 78.17%, top1: 53.78%)                     | JAX  | [Paper](https://arxiv.org/pdf/1409.4842.pdf)  |
| ResNet50    | ImageNet | OK (val-top5: 92.53%, top1: 75.47%)     |  JAX | [Paper](https://arxiv.org/pdf/1512.03385.pdf)  |
| YOLOv1-DarkNet    |  ImageNet | OK (val-top5: 89.39%, top1: 69.1%)     | JAX  | [Paper](https://arxiv.org/pdf/1506.02640.pdf)/[Blog](https://wty-yy.space/posts/1018/)    |
| YOLOv3-DarkNet53  | ImageNet  | OK (val-top5: 92.6%, top1: 75.77%)     |  JAX   | [Paper](https://arxiv.org/pdf/1804.02767.pdf)/[Blog](https://wty-yy.space/posts/50137/)    |
| G-VAE (VAE)    | MNIST,cifar10,celeba  | OK   | JAX   | [Usage](/katacv/G_VAE/)    |
|     |      |      |     |     |

## 目标检测

| 模型      | 数据集   | 完成情况                           | 框架 |      Paper/Note|
| --------- | -------- | ---------------------------------- | ---- |  ----|
| YOLOv1   | VOC PASCAL    | OK (val-mAP: 0.53, coco-mAP: 0.28)    | JAX  | [Paper](https://arxiv.org/pdf/1506.02640.pdf)/[Blog](https://wty-yy.space/posts/1018/)     |
| YOLOv3   | VOC PASCAL <br/> COCO    | OK (PASCAL: val-mAP: 0.66, coco-mAP: 0.39 <br/> COCO: val-mAP: 0.42, coco-mAP: 0.25)    | JAX  | [Paper](https://arxiv.org/pdf/1804.02767.pdf)/[Blog](https://wty-yy.space/posts/50137/)     |

## OCR

| 模型      | 数据集   | 完成情况                           | 框架 |      Paper/Note|
| --------- | -------- | ---------------------------------- | ---- |  ----|
| CTCLoss & CRNN   |  MJsynth   | OK (val accuray: 63 words: 91.23%,<br/>(lower) 37 words: 94.28%)    | JAX  | [CTCLoss Paper](https://www.cs.toronto.edu/~graves/icml_2006.pdf)/[CRNN Paper](https://arxiv.org/pdf/1507.05717.pdf)/[Blog](https://wty-yy.space/posts/62694/)     |

权重参数保存位置（不完整）：[Google网盘-Network Weight](https://drive.google.com/drive/folders/1QfcO48rSF8vZmNhZVBlCgMoQRXN_bUoJ?usp=sharing)

### ImageNet2012 Get Started

对Imagenet2012数据集的解包方法如下：

下载数据集（训练集：ILSVRC2012_img_train.tar，验证集：ILSVRC2012_img_val.tar），并根据 [知乎 - ImageNet(ISLVRC2012)数据集](https://zhuanlan.zhihu.com/p/370799616) 这篇文章对压缩包进行解压并分类。

假设数据集文件保存在 `path/your/imagenet` 下，在该文件夹下分别有 `train/, val/` 文件夹，执行以下代码将数据集转化为tfrecord文件（加快读取速度）：

```shell
python katacv/utils/imagenet/make_label_json.py --path-origin-dataset path/your/imagenet  # 在/logs文件夹下生成两个json文件，对标签进行编号
python katacv/utils/imagenet/translate_tfrecord.py --path-origin-dataset path/your/imagenet --subfolder-name train  # 将train图片转为tfrecord，用时较长约11h
python katacv/utils/imagenet/translate_tfrecord.py --path-origin-dataset path/your/imagenet --subfolder-name val  # 将val图片转为tfrecord
# 生成的tfrecord文件保存在path/your/imagenet/tfrecord文件夹下
python katacv/utils/imagenet/check_tfrecord.py --path-origin-tfrecord path/your/imagenet --subfolder-name train  # 检查train类别中tfrecord正确性
python katacv/utils/imagenet/check_tfrecord.py --path-origin-tfrecord path/your/imagenet --subfolder-name val  # 检查val类别中tfrecord正确性
python katacv/utils/imagenet/build_dataset.py --path-dataset-tfrecord path/your/imagenet  # 建立数据集，直接执行该文件会在训练集中随机采样5个训练图片，并现实其翻译后的标签名
```
