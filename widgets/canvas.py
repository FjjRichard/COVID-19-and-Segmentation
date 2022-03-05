from PyQt5 import QtWidgets
from PyQt5 import QtCore
from PyQt5 import QtGui


class Canvas(QtWidgets.QWidget):
    wheeled = QtCore.pyqtSignal(int)

    def __init__(self, graphicsView):
        super().__init__()
        self.graphicsView = graphicsView
        self.graphicsView.setMinimumSize(QtCore.QSize(512, 512))
        self.graphicsView.setStyleSheet("padding: 0px; border: 0px;background:#000;")
        self.scene = QtWidgets.QGraphicsScene(self)
        self.graphicsView.setAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)  # 改变对齐方式

        self.graphicsView.setSceneRect(0, 0, self.graphicsView.viewport().width(),
                                       self.graphicsView.height())  # 设置图形场景大小和图形视图大小一致
        self.graphicsView.setScene(self.scene)

        self.scene.mousePressEvent = self.scene_mousePressEvent  # 接管图形场景的鼠标按下事件
        self.scene.mouseReleaseEvent = self.scene_mouseReleaseEvent

        self.scene.mouseMoveEvent = self.scene_mouseMoveEvent# 接管图形场景的鼠标移动事件
        self.scene.wheelEvent = self.scene_wheelEvent# 接管图形场景的鼠标滚轮事件

        self.ratio = 1  # 缩放初始比例
        self.zoom_step = 0.1  # 缩放步长
        self.zoom_max = 3.5  # 缩放最大值
        self.zoom_min = 1  # 缩放最小值
        self.pixmapItem = None


        self.currIndex = 0


    def addScenes(self, img, currIndex):  # 绘制图形
        self.currIndex = currIndex
        self.org = img
        self.pixmap = img
        if self.pixmapItem != None:
            originX = self.pixmapItem.x()
            originY = self.pixmapItem.y()
        else:
            originX, originY = 0,0 # 坐标基点
        self.scene.clear()
        self.pixmapItem = self.scene.addPixmap(self.pixmap)
        self.pixmapItem.setScale(self.ratio)  # 缩放
        self.pixmapItem.setPos(originX, originY)

    def scene_mousePressEvent(self, event):
        try:
            if event.button() == QtCore.Qt.LeftButton:  # 左键按下
                pass
            elif event.button() == QtCore.Qt.MidButton:
                self.preMousePosition = event.scenePos()  # 获取鼠标当前位置
                self.pixmapItem.setCursor(QtCore.Qt.SizeAllCursor)
        except Exception as e:
            print(e)

    def scene_mouseReleaseEvent(self,event):
        try:
            if event.button() == QtCore.Qt.MidButton:
                self.pixmapItem.setCursor(QtCore.Qt.ArrowCursor)
        except Exception as e:
            print(e)


    def scene_mouseMoveEvent(self, event):
        if event.buttons() == QtCore.Qt.MidButton:
            self.MouseMove = event.scenePos() - self.preMousePosition  # 鼠标当前位置-先前位置=单次偏移量
            self.preMousePosition = event.scenePos()  # 更新当前鼠标在窗口上的位置，下次移动用
            self.pixmapItem.setPos(self.pixmapItem.pos() + self.MouseMove)  # 更新图元位置


    # 定义滚轮方法。当鼠标在图元范围之外，以图元中心为缩放原点；当鼠标在图元之中，以鼠标悬停位置为缩放中心
    def scene_wheelEvent(self, event):
        """
        滚动鼠标滚轮有两种情况，
        1.单独滚动滚轮，上下翻页
        2.按着右键，再滚动滚轮，放大缩小
        """
        angle = event.delta() / 8
        if event.buttons() == QtCore.Qt.RightButton:
            # 返回QPoint对象，为滚轮转过的数值，单位为1/8度
            if angle > 0:
                # print("滚轮上滚")
                self.ratio += self.zoom_step  # 缩放比例自加
                if self.ratio > self.zoom_max:
                    self.ratio = self.zoom_max
                else:
                    w = self.pixmap.size().width() * (self.ratio - self.zoom_step)
                    h = self.pixmap.size().height() * (self.ratio - self.zoom_step)
                    x1 = self.pixmapItem.pos().x()  # 图元左位置
                    x2 = self.pixmapItem.pos().x() + w  # 图元右位置
                    y1 = self.pixmapItem.pos().y()  # 图元上位置
                    y2 = self.pixmapItem.pos().y() + h  # 图元下位置
                    if event.scenePos().x() > x1 and event.scenePos().x() < x2 \
                            and event.scenePos().y() > y1 and event.scenePos().y() < y2:  # 判断鼠标悬停位置是否在图元中
                        self.pixmapItem.setScale(self.ratio)  # 缩放
                        a1 = event.scenePos() - self.pixmapItem.pos()  # 鼠标与图元左上角的差值
                        a2 = self.ratio / (self.ratio - self.zoom_step) - 1  # 对应比例
                        delta = a1 * a2
                        self.pixmapItem.setPos(self.pixmapItem.pos() - delta)

                    else:
                        # print('在外部')  # 以图元中心缩放
                        self.pixmapItem.setScale(self.ratio)  # 缩放
                        delta_x = (self.pixmap.size().width() * self.zoom_step) / 2  # 图元偏移量
                        delta_y = (self.pixmap.size().height() * self.zoom_step) / 2
                        self.pixmapItem.setPos(self.pixmapItem.pos().x() - delta_x,
                                               self.pixmapItem.pos().y() - delta_y)  # 图元偏移
            else:
                # print("滚轮下滚")
                self.ratio -= self.zoom_step
                if self.ratio < self.zoom_min:
                    self.ratio = self.zoom_min
                else:
                    w = self.pixmap.size().width() * (self.ratio + self.zoom_step)
                    h = self.pixmap.size().height() * (self.ratio + self.zoom_step)
                    x1 = self.pixmapItem.pos().x()
                    x2 = self.pixmapItem.pos().x() + w
                    y1 = self.pixmapItem.pos().y()
                    y2 = self.pixmapItem.pos().y() + h
                    # print(x1, x2, y1, y2)
                    if event.scenePos().x() > x1 and event.scenePos().x() < x2 \
                            and event.scenePos().y() > y1 and event.scenePos().y() < y2:
                        # print('在内部')
                        self.pixmapItem.setScale(self.ratio)  # 缩放
                        a1 = event.scenePos() - self.pixmapItem.pos()  # 鼠标与图元左上角的差值
                        a2 = self.ratio / (self.ratio + self.zoom_step) - 1  # 对应比例
                        delta = a1 * a2
                        self.pixmapItem.setPos(self.pixmapItem.pos() - delta)
                    else:
                        # print('在外部')
                        self.pixmapItem.setScale(self.ratio)
                        delta_x = (self.pixmap.size().width() * self.zoom_step) / 2
                        delta_y = (self.pixmap.size().height() * self.zoom_step) / 2
                        self.pixmapItem.setPos(self.pixmapItem.pos().x() + delta_x, self.pixmapItem.pos().y() + delta_y)
        else:
            try:
                if angle > 0:
                    self.wheeled.emit(self.currIndex + 1)
                elif angle < 0:
                    self.wheeled.emit(self.currIndex - 1)
            except Exception as e:
                print(e)
