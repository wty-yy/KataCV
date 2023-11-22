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
- [x] 完成一院比赛的笔记（晚上）
- [x] 完成对target中wh的分布绘制。（上午）
- [x] 完成PASCAL VOC数据集的读取工作。（下午）
- [x] 利用orbax实现对模型权重的保存，从而废弃掉之前的旧保存方法，可以在读取权重前提下，不对模型进行初始化；考虑创建新的文件katacv.utils.weights，在其中对load_weights和SaveWeightsManager用orbax进行重写（下午）
- [x] 开始对PASCAL VOC数据集进行训练。（下午）
- [x] 晚上记得上人与环境（晚上）

### 2023/11/21
在PASCAL VOC数据集上训练了100个epochs用时5h，最终val_loss=15，train_loss=10左右，识别效果依然非常差，思考原因：
1. 首先即使在训练集上，边界框的框选大小均偏大，CIOU损失的问题？
2. human的识别准确率和置信度非常低，这个可能是因为没有加入ignore样本

修改：
1. 首先将DIOU和CIOU都写成和IOU正相关的形式，也就是不减去1（代码设计问题），注意计算ciou的loss时候最后做mask，不然会导致loss的显示错误（不影响梯度计算）。
2. 将目标框的锚框选择从IOU转为CIOU（有小影响）。
3. 加入ignore_threshold，对于每个预测框，只要和当前的ground truth中的任意一个的IOU > ignore_threthold，
   则不将其加入到noobj中，也就是不对其Pr(obj)进行减小，这样应该可以避免预测置信度过小的情况。（重大影响）

BUGs:
1. 计算CIOU中，正则项系数alpha忘记`stop_gradient`了，这可能是导致边界框大小错误的原因。（重大影响）
2. 计算CIOU时候，必须对wh乘以anchors之后再进行计算，因为wh的比例不同，且也会对DIOU中c^2产生影响。（重大影响，边界框位置基本都不准确）

本次修改后：损失下降速度慢很多，尝试将`base_learning_rate`调整为`1e-3`（原来为`2.5e-4`）

重新加入loss权重系数：
- 设置`noobj`和`coord`的系数为2，剩余`obj`和`class`的系数为1。
- 重新设置 `obj=class=2,noobj=1,coord=0.5`

尝试换成原来的写法，用bce预测xy，用mse预测wh，需要修改以下三个部分
- `single_loss_fn`中的`loss_coord`改为bce+mse
- `build_target`中对wh除以`anchors[j,k,0/1]`
- 减小学习率`base_learning_rate=2.5e-4`
- 重新设置损失系数`noobj=coord=2.0,obj=class=1.0`


- [x] 完成YOLOv4的进一步修正（上午）
- [x] 重新在PASCAL VOC上训练5h（中午）
- [ ] 设计高效求解mAP方法，也就是求PR曲线（中午，下午实现）
- [ ] 标记100帧数据集（下午）
- [ ] 阅读一片论文（下午）

### 2023/11/22
使用MSE作为wh的损失，还是用之前的做法，xy作用sigmoid，wh作用exp函数，
昨天忘记将