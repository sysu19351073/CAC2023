import copy
import numpy as np
import cv2
from utils import *
from visualAngle import *
import time
from findColors import *
from PIL import Image

class FaceCamera():
    def __init__(self, camera_type, camera_index, botton, visual_index, vertical_flip=False, horizontal_flip=False):
        # 朝向魔方的位置
        self.camera_type = camera_type

        # 镜头编号
        self.camera_index = camera_index

        # 控制按键
        self.botton = botton

        # 摄像头参数
        # VideoCapture()中参数是0，表示打开笔记本的内置摄像头；
        # CAP_DSHOW参数可以解决部分摄像头无法打开的问题，但并不适用于所有情况
        # 使用CAP_DSHOW参数打开摄像头DirectShow
        self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
        # cv2.VideoCapture().set(propId, value)；
        # propId：设置的视频参数，3：在视频流的帧的宽度，4，在视频流的帧的高度；value：设置的参数数值
        self.cap.set(3, 500)
        self.cap.set(4, 500)
        # （FALSE）不进行垂直水平方向反转
        self.vertical_flip = vertical_flip
        self.horizontal_flip = horizontal_flip

    def process(self):

        while (True):
            # 摄像头读取数据流，返回视频是否结束，和每一帧的图像，frame
            _, frame = self.cap.read()

            # 调整摄像机视角
            # cv2.flip(filename, flipcode) filename：需要操作的图像；flipcode：翻转方式
            # 1：水平翻转；0：垂直翻转；-1：水平垂直翻转
            if self.vertical_flip:
                frame = cv2.flip(frame, 0)
            if self.horizontal_flip:
                frame = cv2.flip(frame, 1)

            # 浅拷贝
            frame_show = copy.copy(frame)

            # 在窗口frame_show_{} 显示图像frame_show，{}中为魔方的面（摄像头朝向）
            cv2.imshow('frame_show_{}'.format(self.camera_type), frame)

            key = cv2.waitKey(50)

            # 此面捕捉完毕，按 self.botton 完成
            # ord（true）= 1
            if key == ord(self.botton):
                # 保存此处镜头画面
                # 得到该面9个小色块的BGR
                face_colors = []
                num = 0
                cv2.imwrite('temp_camera/camera_frame_ori{}.png'.format(self.camera_type), frame)
                if self.camera_type == 'U' or self.camera_type == 'F' or self.camera_type =='D':
                    images = cv2.transpose(frame)
                    images = cv2.flip(images, 0)
                else:
                    images = frame
                # cv2.imwrite('temp_camera/camera_frame{}.png'.format(self.camera_type), frame_show)
                # 第一种方法：定位截取颜色
                # face_colors = self.visualAngle.getColors(frame)
                # 第二种方法：图像处理获得色块

                num, face_colors = findColors(images)
                # 第三种方法：检测中心点
                # face_colors = findColors2(frame, center_restrain)


                # for color in face_colors:
                #     if color[0] > 1:
                #         # color[0] = 0
                #         num = num + 1
                if num < 9:
                    for color in face_colors:
                            color[0] = 0
                    print('重新捕捉第{}面'.format(self.camera_type))
                else:
                    self.cap.release()
                    print('完成第{}面的获取'.format(self.camera_type))
                    colors = face_colors.copy()
                    for color in face_colors:
                            color[0] = 0
                    return colors
                    break



        # 有些视角需要旋转处理
        # if self.camera_type == 'U':
        #     face_colors2 = np.empty((9,3))
        #     # 直接赋值，不用循环，可能会快些
        #     face_colors2[0] = face_colors[2]
        #     face_colors2[1] = face_colors[5]
        #     face_colors2[2] = face_colors[8]
        #     face_colors2[3] = face_colors[1]
        #     face_colors2[4] = face_colors[4]
        #     face_colors2[5] = face_colors[7]
        #     face_colors2[6] = face_colors[0]
        #     face_colors2[7] = face_colors[3]
        #     face_colors2[8] = face_colors[6]
        #     face_colors = face_colors2

