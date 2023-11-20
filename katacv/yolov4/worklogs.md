### 2023/11/18
完成YOLOv4的全部代码部分，完成小demo上的训练测试，开始coco上训练。

### 2023/11/20
发现loss完全不收敛，发现问题应该出在学习率过大的问题上，目前设置为1e-4，（也有可能是数据增强过大，可以考虑只保留图像色彩变化和左右翻转）

**发现严重bug**，数据集的输出为COCO类型但是我将其当作YOLO中心点直接进行处理了（WTF）。

当前的训练结果：loss_val：稳定在48附近，loss_train：稳定在41附近，估计必须将loss训练到10左右才能有使用。

学习其他模型的学习率设定方法：
- pytorch-YOLOv4：使用batchsize=64，lr=0.00261/batchsize=4e-5, epochs=300
- YouTube-YOLOv3: 使用batchsize=32, lr=1e-5, epochs=100

- flax-imagenet-resnet: 给出一个最大lr和batchsize的关系式：base_learning_rate = config.learning_rate * config.batch_size / 256.0, config.learning_rate=0.1
综上来看，将学习率设置为1e-3和1e-4都明显过大，应该设置到3e-5左右，考虑使用flax-imagenet的关系式，将config.learning_rate=2.5e-4，进行重新训练300个epochs。

> 一个epochs用时22mins，预计花费110h，也就是4.6天完成训练，所以考虑在服务器上训练，并且要求对模型权重数据只保留最后10个，避免内存溢出，预计要使用。

由于我们只想测试YOLOv4的性能，所以可以现在PASCAL VOC这个小数据集上进行测试，预估使用100个epochs应该就能完成训练。这样估计1天能够完成（训练集大小只有16551）
由于这里学习率本身就很低，所以warmup_epoch可以设置较小，考虑为2.

所以我们下一步要做的是完成对PASCAL VOC2004/2007数据集的读取工作，可以考虑直接用之前处理好的，将labels从原来的yolo标签转化为COCO格式就能够直接使用COCO的读取方式进行读取了。

- [x] 完成YOLOv4的笔记（上午及下午）
- [ ] 完成一院比赛的笔记（晚上）
- [x] 完成对target中wh的分布绘制。（上午）
- [x] 完成PASCAL VOC数据集的读取工作。（下午）
- [x] 利用orbax实现对模型权重的保存，从而废弃掉之前的旧保存方法，可以在读取权重前提下，不对模型进行初始化；考虑创建新的文件katacv.utils.weights，在其中对load_weights和SaveWeightsManager用orbax进行重写（下午）
- [ ] 开始对PASCAL VOC数据集进行训练。（下午）
- [ ] 晚上记得上人与环境（晚上）

