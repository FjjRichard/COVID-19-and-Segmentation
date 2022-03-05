# -*- coding: utf-8 -*-
import cv2

import numpy as np
import paddle
import paddleseg.transforms as T
from paddleseg.core import infer

import SimpleITK as sitk

def cnt_area(cnt):
    """返回轮廓的面积"""
    area = cv2.contourArea(cnt)
    return area

def wwwc(sitkImage, ww=1500, wc=-550):
    # 设置窗宽窗位
    min = int(wc - ww / 2.0)
    max = int(wc + ww / 2.0)
    intensityWindow = sitk.IntensityWindowingImageFilter()
    intensityWindow.SetWindowMaximum(max)
    intensityWindow.SetWindowMinimum(min)
    sitkImage = intensityWindow.Execute(sitkImage)
    return sitkImage


def readNii(path, ww, wc, isflipud=True, ):
    """读取和加载数据"""
    if type(path) == str:
        img = wwwc(sitk.ReadImage(path), ww, wc)
    else:
        img = wwwc(path, ww, wc)
    data = sitk.GetArrayFromImage(img)
    # 图像是上下翻转的，所有把他们翻转过来
    # 不知道为什么这个肺部CT数据用SimpleITK读取是上下翻转的。
    if isflipud:
        data = np.flip(data, 1)
    return data



def nn_infer(model, im):
    # 预测结果
    transforms = T.Compose([
        T.Resize(target_size=(512, 512)),
        T.Normalize()
    ])
    img, _ = transforms(im)
    img = paddle.to_tensor(img[np.newaxis, :])
    pre = infer.inference(model, img)
    pred = paddle.argmax(pre, axis=1).numpy().reshape((512, 512))
    return pred.astype('uint8')


