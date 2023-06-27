import numpy as np
import cv2
from utils import *
from solveCube import *
import matplotlib.pyplot as plt
from cameraProcess import *
from k_means import *
import serial
import time

# 串口通讯
serialPort = "COM4"  # 串口
baudRate = 115200  # 波特率
ser = serial.serial(serialPort, baudRate, timeout=0.5)
print("参数设置：串口=%s ，波特率=%d" % (serialPort, baudRate))

# U R F D L B
# 调试用的预设值
np.set_printoptions(suppress=True)

# 摄像头编号
F_index = 3
B_index = 1
L_index = 4
U_index = 2

# 存放六个面共54个色块的bgr像素值
all_colors = np.empty((54,3))

time.sleep(5)  # 初始化需要2s

scanFlag = False
solveFlag = True

ser.write('f'.encode())

# 扫描
if scanFlag:
    ser.write('f'.encode())

    while(1):
        str = ser.readlines()
        print(str)
        # if len(str) is not 0:
        #     command = str[0]
        if b'ScanOver\r\n' in str:
            break
        elif b'Scan1\r\n' in str:
            camera1 = FaceCamera2('U', U_index, 1)
            all_colors[0:9] = camera1.process()  # U
            camera2 = FaceCamera2('L', L_index, 1)
            all_colors[36:45] = camera2.process()  # L
            camera3 = FaceCamera2('F', F_index, 2)
            all_colors[18:27] = camera3.process()  # F
            camera4 = FaceCamera2('B', B_index, 3)
            all_colors[45:54] = camera4.process()  # B
            # 此时F和B只拍到8个色块
            # 左：松爪 右：转90度
        elif b'Scan2\r\n' in str:
            camera = FaceCamera2('F', F_index, 2)  # 和F是同一个摄像头
            all_colors[27:36] = camera.process()  # D
            # 右：反向90度转回来
            # 左握紧，右松手：再拍一次F和B
        elif b'Scan3\r\n' in str:
            camera = FaceCamera2('F', F_index, 2)
            all_colors[23] = camera.process()[5]  # F
            camera = FaceCamera2('B', B_index, 3)
            all_colors[48] = camera.process()[3]  # B
            # 左：转90度
        elif b'Scan4\r\n' in str:
            camera = FaceCamera2('F', F_index, 2)  # 和F是同一个摄像头
            all_colors[9:18] = camera.process()  # R
            # 左：反向90度转回来
        else:
            time.sleep(1)

    # print(all_colors)


    centers_color = all_colors[4:54:9]  # 6个中心点颜色

    # 聚类
    cube_position = k_means(all_colors, centers_color)
    print('聚类结果:', cube_position)

    centers_cls = cube_position[4:54:9]  # 6个中心点颜色类别

    # 可视化色块
    # 将六个面分开
    six_face_colors = all_colors.reshape(6,-1,3)
    # face_colors大小为 (9*3)
    for face_index,face_colors in enumerate(six_face_colors):
        face_cls = cube_position[face_index*9:(face_index+1)*9]
        showColor(face_index,face_colors,face_cls)

    # 检测魔方是否有效
    if cubeIsValid(cube_position, centers_cls):
        print('魔方有效')
        # 复原
        results = cubeSolve(cube_position, centers_cls)
        print('魔方复原公式：', results)
        # 发送至 Arduino

    else:
        print('魔方无效')


# results = 'F1 D2 L2 B3 D3 R1 L3 U1 R1 L3 F3 L1 F1 B1 L2 U2 B1 U2 F1 U2 '
# results = 'F3 L1 F1 B1 L2 U2 B1 U2 F1 U2 '
results = 'F3 L1 '
# 将魔方公式通过串口输出
# time.sleep(1)

# 还原
if solveFlag:
    ser.write('h'.encode())
    ser.write(results.encode())

    while(1):
        str = ser.readlines()
        print(str)
        time.sleep(1)







