# G-VAE (Generative VAE)

概述：传统VAE只能用于重建，现在将其中中加入分类模型，并利用[ISDA Loss](https://zhuanlan.zhihu.com/p/344953635?utm_id=0)作为损失函数，能够有效地在同种类别上进行数据增强。

## 效果演示

中间列为原图，左右列分别为按照两个方向的增强结果，步长为 0.5sigma）

**MNIST**（左G-VAE，右VAE）

<div style="display: flex;">
  <div style="flex: 49%; padding: 5px; display: flex; justify-content: center; align-items: center;">
    <img src="../../archives/figures/G-VAE_MNIST_aug.jpg" alt="G-VAE对MNIST进行数据增强">
  </div>
  <div style="flex: 49%; padding: 5px; display: flex; justify-content: center; align-items: center;">
    <img src="../../archives/figures/VAE_MNIST_aug.jpg" alt="VAE对MNIST进行数据增强">
  </div>
</div>

**cifar10**（左G-VAE，右VAE，最右列为加入Gauss噪声后的增强结果）

<div style="display: flex;">
  <div style="flex: 49%; padding: 5px; display: flex; justify-content: center; align-items: center;">
    <img src="../../archives/figures/G-VAE_cifar10_aug.jpg" alt="G-VAE对cifar10进行数据增强">
  </div>
  <div style="flex: 49%; padding: 5px; display: flex; justify-content: center; align-items: center;">
    <img src="../../archives/figures/VAE_cifar10_aug.jpg" alt="VAE对cifar10进行数据增强">
  </div>
</div>

## 代码框架

解释每个代码的具体功能：

1. [`katacv/G_VAE/g_vae.py`](/katacv/G_VAE/g_vae.py), [`katacv/G_VAE/vae.py`](/katacv/G_VAE/g_vae.py)：分别为G-VAE和VAE的训练代码，包含参数解析、日志管理、创建模型、加载权重、权重管理、数据集读入、模型训练。
2. 