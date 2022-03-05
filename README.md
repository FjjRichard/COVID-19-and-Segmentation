# 基于PaddleSeg分割毛玻璃病灶并进行可视化

## 项目描述

#### ### 新冠肺炎一般是肺部出现磨玻璃样病灶，现在使用[PaddleSeg](https://github.com/PaddlePaddle/PaddleSeg)训练语义分割模型，对胸部CT上的毛玻璃病灶进行分割。使用PYQT搭建UI界面，对分割结果可视化。

## 项目结构

```
./data  #测试数据
./model #分割模型
./utils 
    utils.py 
./widgets 
    canvas.py #PyQt控件代码
main.py #程序主文件
SegGroundClassUI.py #UI代码文件
SegGroundClassUI.ui
requirements.txt
```

## 快速开始

1. ``pip install` `-``r requirements.txt``

2. 加载测试数据，加载分割模型，开始推理
   
   ![](C:\Users\Richard\Desktop\捕1.png)



## 提示

1. 按鼠标中间可以移动图像

2. 按住鼠标右键，滚动滑轮可以放大、缩小

3. [B站展示地址](https://www.bilibili.com/video/BV1ag411K7A7?share_source=copy_web)



## 模型训练方法

A：在AI Studio上[运行本项目](https://aistudio.baidu.com/aistudio/projectdetail/2574999)