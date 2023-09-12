# KataCV

> 原replicate-papers改名为KataCV，用于复现经典CV文献，当前用于个人测试

## 图像分类

| 网络      | 数据集   | 完成情况                           | 框架 |
| --------- | -------- | ---------------------------------- | ---- |
| LeNet-5   | MNIST    | OK                                 | TF2  |
| AlexNet   | ImageNet | OK 参数在网盘中                    | TF2  |
| VGG16     | ImageNet | OK（但是调参仍有问题，初始lr过大） | JAX  |
| GoogleNet | ImageNet | OK（训练中）                       | JAX  |

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

## 目标识别

| 算法    | 数据集     | 完成情况 | 框架 |
| ------- | ---------- | -------- | ---- |
| YOLO-v1 | Pascal VOC | working  | JAX  |
|         |            |          |      |
|         |            |          |      |

