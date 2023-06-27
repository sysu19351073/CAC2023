import cv2
import numpy as np


class visualAngle1():
    def __init__(self):
        # # 框选参数
        # self.xmin = 170  # 左上角框中心点 x坐标
        # self.ymin = 75  # 左上角框中心点 y坐标
        # self.x_dis = 150  # 相邻框中心点的横向偏移距离
        # self.y_dis = 120  # 相邻框中心点的纵向偏移距离
        # self.rec_size = 20  # 框边长的一半

        # 用列表来存储位置信息
        # 这里可以个性化每个视角
        # self.positionList = []
        #
        # for i in range(3):
        #     for j in range(3):
        #         # 从左上开始，每行从左到右，到右下
        #         z = self.xmin + j * self.x_dis
        #         y = self.ymin + i * self.y_dis
        #         self.positionList.append(
        #             [(z - self.rec_size, y - self.rec_size), (z + self.rec_size, y + self.rec_size)])

        self.positionList = [
            [(160, 55), (215, 110)],
            [(300, 55), (355, 110)],
            [(430, 55), (485, 110)],
            [(160, 175), (215, 230)],
            [(300, 175), (355, 230)],
            [(430, 175), (485, 230)],
            [(160, 295), (215, 350)],
            [(300, 295), (355, 350)],
            [(430, 295), (485, 350)],
        ]

        # 存储结果，大小为(9,3)
        #
        self.face_colors = np.empty((9, 3))

    def showAngle(self, frame_show):
        for i in range(9):
            # cv2.rectangle(img, pt1, pt2, color, thickness, lineType, shift )
            # 参数表示依次为： （图片，长方形框左上角坐标, 长方形框右下角坐标， 字体颜色，字体粗细）
            cv2.rectangle(frame_show, (self.positionList[i][0][0], self.positionList[i][0][1]),
                          (self.positionList[i][1][0], self.positionList[i][1][1]),
                          (150, 150, 150), 3)

    def getColors(self, frame):
        # enumerate（）函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列，同时列出数据和数据下标
        for i, position in enumerate(self.positionList):
            target = frame[position[0][1]:position[1][1], position[0][0]:position[1][0],
                     :]  # 注意这里xy反过来，因为frame形状是高*宽
            # print(z-rec_size,z+rec_size,y-rec_size,y+rec_size)
            # print('色块{}：中心点{} {}'.format(i, z, y))

            # 处理色块：求框选区域的均值
            target = np.squeeze(np.mean(np.mean(target, axis=1), axis=0))
            self.face_colors[i, :] = target  # 第 i 个色块
        return self.face_colors  # 得到(9,3)

class visualAngle2():
    def __init__(self):
        # 用列表来存储位置信息
        # 这里可以个性化每个视角

        self.positionList = [
            [(160, 55), (215, 110)],
            [(300, 55), (355, 110)],
            [(430, 75), (485, 130)],
            [(160, 175), (215, 230)],
            [(300, 175), (355, 230)],
            [(430, 195), (485, 250)],
            [(160, 295), (215, 350)],
            [(300, 295), (355, 350)],
            [(430, 315), (485, 370)],
        ]

        # 存储结果，大小为(9,3)
        self.face_colors = np.empty((9, 3))

    def showAngle(self, frame_show):
        for i in range(9):
            cv2.rectangle(frame_show, (self.positionList[i][0][0], self.positionList[i][0][1]),
                          (self.positionList[i][1][0], self.positionList[i][1][1]),
                          (150, 150, 150), 3)

    def getColors(self, frame):
        for i, position in enumerate(self.positionList):
            target = frame[position[0][1]:position[1][1], position[0][0]:position[1][0],
                     :]  # 注意这里xy反过来，因为frame形状是高*宽q

            # 处理色块：求框选区域的均值
            target = np.squeeze(np.mean(np.mean(target, axis=1), axis=0))
            self.face_colors[i, :] = target  # 第 i 个色块
        return self.face_colors  # 得到(9,3)

class visualAngle3():
    def __init__(self):
        # 用列表来存储位置信息
        # 这里可以个性化每个视角

        self.positionList = [
            [(180, 55), (235, 110)],
            [(300, 55), (355, 110)],
            [(430, 75), (485, 130)],
            [(160, 175), (215, 230)],
            [(300, 175), (355, 230)],
            [(430, 195), (485, 250)],
            [(160, 295), (215, 350)],
            [(300, 295), (355, 350)],
            [(430, 315), (485, 370)],
        ]

        # 存储结果，大小为(9,3)
        self.face_colors = np.empty((9, 3))

    def showAngle(self, frame_show):
        for i in range(9):
            cv2.rectangle(frame_show, (self.positionList[i][0][0], self.positionList[i][0][1]),
                          (self.positionList[i][1][0], self.positionList[i][1][1]),
                          (150, 150, 150), 3)

    def getColors(self, frame):
        for i, position in enumerate(self.positionList):
            target = frame[position[0][1]:position[1][1], position[0][0]:position[1][0],
                     :]  # 注意这里xy反过来，因为frame形状是高*宽q

            # 处理色块：求框选区域的均值
            target = np.squeeze(np.mean(np.mean(target, axis=1), axis=0))
            self.face_colors[i, :] = target  # 第 i 个色块
        return self.face_colors  # 得到(9,3)