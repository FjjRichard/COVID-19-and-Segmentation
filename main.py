# -*- coding: utf-8 -*-
# pyuic5 -o SegGroundClassUI.py SegGroundClassUI.ui
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from SegGroundClassUI import Ui_MainWindow

from paddleseg.models import BiSeNetV2, UNet
import paddleseg.transforms as T
from paddleseg.core import infer
import paddle
import numpy as np
import SimpleITK as sitk
import os
import cv2
from utils.utils import readNii
from widgets.canvas import Canvas

class InferThread(QThread):
    """
    建立一个任务线程类,  推理任务
    """
    signal_infer_fail = pyqtSignal()  # 推理失败的信号
    signal_infer_result = pyqtSignal(np.ndarray)  # 这信号用来传递推理结果

    def __init__(self, sitkImage, model):
        super(InferThread, self).__init__()
        self.sitkImage = sitkImage
        self.model = model
        self.transforms = T.Compose([
            T.Resize(target_size=(512, 512)),
            T.Normalize()
        ])

    def run(self):  # 在启动线程后任务从这个函数里面开始执行
        try:
            data = readNii(self.sitkImage, 1500, -500)
            inferData = np.zeros_like(data)
            d, h, w = data.shape

            for i in range(d):
                img = data[i].copy()
                img = img.astype(np.float32)
                pre = self.nn_infer(self.model, img, self.transforms)
                inferData[i] = pre

            self.signal_infer_result.emit(inferData)
        except Exception as e:
            print(e)
            self.signal_infer_fail.emit()

    def nn_infer(self, model, im, transforms):
        # 预测结果

        img, _ = transforms(im)
        img = paddle.to_tensor(img[np.newaxis, :])
        pre = infer.inference(model, img)
        pred = paddle.argmax(pre, axis=1).numpy().reshape((512, 512))
        return pred.astype('uint8')


class MainWindow(QMainWindow,Ui_MainWindow):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setupUi(self)
        self.canvas = Canvas(self.graphicsView)
        self.canvas.wheeled.connect(self.myWheel)


        self.initUI()
        self.setWindowTitle('结合PYQT5部署PaddleSeg模型')
        self.bn_open.clicked.connect(self.openFile)  # 打开 nii文件选择器
        self.bn_loadModel.clicked.connect(self.openModleFile)  # 打开模型文件选择器
        self.bn_infer.clicked.connect(self.infer)  # 推理按钮
        self.bn_output.clicked.connect(self.outputFile)

        self.sitkImage = object()
        self.npImage = object()
        self.currIndex = 0  # 记录当前第几层
        self.maxCurrIndex = 0  # 记录数据的最大层
        self.minCurrIndex = 0  # 记录数据的最小层，其实就是0
        self.baseFileName = ''
        self.isInferSucceed = False


        self.model = object()
        # 判断模型是否加载成功
        self.isModelReady = False

        # 宽宽窗位滑动条
        self.slider_ww.valueChanged.connect(self.resetWWWcAndShow)
        self.slider_wc.valueChanged.connect(self.resetWWWcAndShow)
        # 窗宽窗位下来框选择器
        self.cb_wwwc.currentIndexChanged.connect(self.resetWWWcAndShow)

        # 设置窗宽窗位文本框只能输入一定范围的整数
        intValidator = QIntValidator(self)
        intValidator.setRange(-2000, 2000)
        self.line_ww.setValidator(intValidator)
        self.line_ww.editingFinished.connect(self.resetWWWcAndShow)
        self.line_wc.setValidator(intValidator)
        self.line_wc.editingFinished.connect(self.resetWWWcAndShow)

        self.listWidget.itemDoubleClicked.connect(self.changeLayer)

    def initUI(self):
        try:
            self.wwwcList = {'软组织窗': [350, 80],
                             "纵隔窗": [300, 40],
                             "脑窗": [100, 40],
                             '肺窗': [1700, -700],
                             '骨窗': [1400, 350]}

            windowWidth = self.cb_wwwc.currentText()
            self.line_ww.setText(str(self.wwwcList[windowWidth][0]))
            self.line_wc.setText(str(self.wwwcList[windowWidth][1]))

            self.slider_ww.setValue(self.wwwcList[windowWidth][0])
            self.slider_wc.setValue(self.wwwcList[windowWidth][1])
            self.ww = self.wwwcList[windowWidth][0]
            self.wc = self.wwwcList[windowWidth][1]

            self.currWw = self.ww
            self.currWc = self.wc
        except Exception as e:
            print(e)


    def openFile(self):
        """
        打开医学影像文件选择器
        """
        try:
            filename, _ = QFileDialog.getOpenFileName(self,
                                                      "选取文件",
                                                      "./",
                                                      "Nii Files (*.nii);;Nii Files (*.nii.gz);;All Files (*)")
            if filename:
                self.listWidget.clear()  # 清空列表
                self.isInferSucceed = False
                self.text_loadModel.setText("数据加载成功！")

                self.baseFileName = os.path.basename(filename).split('.')[0]
                self.sitkImage = sitk.ReadImage(filename)

                self.npImage = readNii(self.sitkImage, self.ww, self.wc)
                print(self.npImage.shape)
                self.maxCurrIndex = self.npImage.shape[0]
                self.currIndex = int(self.maxCurrIndex / 2)
                self.showImg(self.npImage[self.currIndex])
        except Exception as e:
            print(e)

    def openModleFile(self):
        """
        打开模型文件选择器
        """
        filename, _ = QFileDialog.getOpenFileName(self, "选取文件", "./", "model Files (*.pdparams)")

        if filename:
            try:
                self.isInferSucceed = False
                self.text_loadModel.setText(" ")
                model_name = self.cb_modelList.currentText()
                num_class = int(self.spinBox_numClass.value())
                if model_name == "BiSeNetV2":
                    self.model = BiSeNetV2(num_classes=num_class)
                elif model_name == "UNet":
                    self.model = UNet(num_classes=num_class)
                para_state_dict = paddle.load(filename)
                self.model.set_dict(para_state_dict)
                self.text_loadModel.setText("模型加载成功！")
                self.isModelReady = True
            except Exception as e:
                self.text_loadModel.setText("模型加载失败！")
                print(e)


    def showImg(self, img):
        """
        显示图片
        """
        try:
            if img.ndim == 2:
                img = np.expand_dims(img, axis=2)
                img = np.concatenate((img, img, img), axis=-1).astype(np.uint8)
            elif img.ndim == 3:
                img = img.astype(np.uint8)
            qimage = QImage(img, img.shape[0], img.shape[1], img.shape[1] * 3, QImage.Format_RGB888)
            pixmap_imgSrc = QPixmap.fromImage(qimage)

            self.canvas.addScenes(pixmap_imgSrc,self.currIndex)
        except Exception as e:
            print(e)

    def resetWWWcAndShow(self):
        """
        有四个方式可以修改医学图像的窗宽窗位，
        每次修改后都会在界面呈现出来
        """
        if hasattr(self.sender(), "objectName"):
            objectName = self.sender().objectName()
        else:
            objectName = None
        try:

            if objectName == 'cb_wwwc':
                windowWidth = self.cb_wwwc.currentText()
                self.line_ww.setText(str(self.wwwcList[windowWidth][0]))
                self.line_wc.setText(str(self.wwwcList[windowWidth][1]))
                self.slider_ww.setValue(self.wwwcList[windowWidth][0])
                self.slider_wc.setValue(self.wwwcList[windowWidth][1])
                self.ww = self.wwwcList[windowWidth][0]
                self.wc = self.wwwcList[windowWidth][1]
                self.currWw = self.ww
                self.currWc = self.wc
            elif objectName == 'slider_ww' or objectName == 'slider_wc':
                self.currWw = self.slider_ww.value()
                self.currWc = self.slider_wc.value()
                self.line_ww.setText(str(self.currWw))
                self.line_wc.setText(str(self.currWc))
            elif objectName == 'line_ww' or objectName == 'line_wc':
                self.currWw = int(self.line_ww.text())
                self.currWc = int(self.line_wc.text())
                self.slider_ww.setValue(self.currWw)
                self.slider_wc.setValue(self.currWc)
            if self.maxCurrIndex != self.minCurrIndex:
                self.npImage = readNii(self.sitkImage, self.currWw, self.currWc)
                if self.isInferSucceed:
                    self.showImg(self.drawContours(self.npImage, self.inferData, self.currIndex))
                else:
                    self.showImg(self.npImage[self.currIndex])
        except Exception as e:
            print(e)

    def infer(self):
        """
        模型分割预测
        """
        if self.maxCurrIndex != self.minCurrIndex and self.isModelReady:
            self.bn_infer.setEnabled(True)
            # 创建推理线程
            self.infer_thread = InferThread(self.sitkImage, self.model)
            # 绑定推理失败的槽函数
            self.infer_thread.signal_infer_fail.connect(self.infer_fail)
            # 绑定推理成功的槽函数
            self.infer_thread.signal_infer_result.connect(self.infer_result)
            self.infer_thread.start()
            self.text_loadModel.setText("模型推理中！")

        else:
            QMessageBox.warning(self, "警告", "请加载模型或者加载数据再进行推理", QMessageBox.Yes, QMessageBox.Yes)

    def infer_result(self, inferData):
        """
        分割模型预测成功后，结果保存在self.inferData
        """
        # 推理成功，并显示结果
        try:
            self.inferData = inferData.astype(np.uint8)
            QMessageBox.information(self, "信息", "推理完成！", QMessageBox.Yes, QMessageBox.Yes)
            self.text_loadModel.setText("模型推理成功！")
            self.isInferSucceed = True
            self.infer_thread.quit()
            self.addListInfo(self.inferData)
            self.showImg(self.drawContours(self.npImage, self.inferData, self.currIndex))
        except Exception as e:
            print(e)

    def infer_fail(self):
        """
        推理失败的情况
        """
        QMessageBox.warning(self, "警告", "推理失败！", QMessageBox.Yes, QMessageBox.Yes)

    def outputFile(self):
        """
        保存模型预测结果为nii格式文件
        """
        try:
            if self.isInferSucceed:
                filedir = QFileDialog.getExistingDirectory(None, "文件保存", os.getcwd())
                if filedir:
                    #因为读取nii文件转换np文件时对数据进行上下翻转，才输入模型推理的
                    #所有现在保存回nii文件，要翻转回来。
                    #不知道为什么这个肺部CT数据用SimpleITK读取是上下翻转的。
                    self.inferData = np.flip(self.inferData, 1)
                    pre_sitkImage = sitk.GetImageFromArray(self.inferData)
                    pre_sitkImage.CopyInformation(self.sitkImage)
                    pre_sitkImage = sitk.Cast(pre_sitkImage, sitk.sitkUInt8)
                    save_path = os.path.join(filedir, self.baseFileName + '_mask.nii')
                    sitk.WriteImage(pre_sitkImage, save_path)
            else:
                QMessageBox.warning(self, "警告", "无进行过推理，无法保存！", QMessageBox.Yes, QMessageBox.Yes)
        except Exception as e:
            print(e)

    def drawContours(self, npImage, inferData, currIndex):
        """
        把mask转换成轮廓绘制在原图上
        """
        img = npImage[currIndex]
        img = np.expand_dims(img, axis=2)
        img = np.concatenate((img, img, img), axis=-1).astype(np.uint8)
        ret, thresh = cv2.threshold(inferData[currIndex], 0, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, kernel=np.ones((5, 5), np.uint8), iterations=1)
        contours, hierarchy = cv2.findContours(thresh, 1, 2)
        # 这是画轮廓
        img = cv2.drawContours(img, contours, -1, (0, 255, 0), 1)

        return img

    def addListInfo(self, inferData):
        """
        增加列表信息
        """
        self.listWidget.clear()
        d, h, w = inferData.shape
        result = {}
        for i in range(d):
            img = inferData[i]
            if np.sum(img > 0) != 0:
                result[str(i)] = np.sum(img > 0)

        result = sorted(result.items(), key=lambda x: x[1], reverse=True)
        for key, value in result:
            self.listWidget.addItem("层 " + str(int(key) + 1))

    def changeLayer(self, item):
        """点击列表自动展示该层"""
        self.currIndex = int(item.text().split(' ')[1]) - 1
        if self.isInferSucceed:
            self.showImg(self.drawContours(self.npImage, self.inferData, self.currIndex))
        else:
            self.showImg(self.npImage[self.currIndex])


    """画布事件"""
    def myWheel(self,currIndex):
        """
        滚轮切换上下层
        """
        try:
            if self.maxCurrIndex != self.minCurrIndex:
                if currIndex > self.maxCurrIndex-1 :
                    self.currIndex = self.maxCurrIndex-1
                elif currIndex < 0:
                    self.currIndex = 0
                else:
                    self.currIndex = currIndex
                if self.isInferSucceed:
                    # self.npImage = self.drawContours(self.npImage, self.inferData)
                    self.showImg(self.drawContours(self.npImage, self.inferData, self.currIndex))
                else:
                    self.showImg(self.npImage[self.currIndex])
        except Exception as e:
            print(e)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())
