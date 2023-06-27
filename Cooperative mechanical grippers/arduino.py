import numpy as np
import cv2
from utils import *
from solveCube import *
import matplotlib.pyplot as plt
from cameraProcess import *
from k_means import *
import serial
import time


# # 串口通讯
# serialPort = "COM6"  # 串口
# baudRate = 9600  # 波特率
# ser = serial.Serial(serialPort, baudRate, timeout=1)
# print("参数设置：串口=%s ，波特率=%d" % (serialPort, baudRate))
#
#
# # while(1):
# #     command = input()
# #     if command == 'z':
# #         break
# # time.sleep(2)
#
# while(1):
#     start = ser.readline()
#     if start != b'':
#         start = start.split()[0].decode()
#         print(start)
#         if start == "OK":
#             break
#
# time.sleep(1)

# U R F D L B
# 调试用的预设值
# #控制Python中小数的显示精度；suppress：小数是否需要以科学计数法的形式输出
np.set_printoptions(suppress=True)

# 摄像头编号
# 3 1 4 2
F_index = 1
B_index = 2
L_index = 2
U_index = 1

# # 存放六个面共54个色块的rbg像素值
all_colors = np.empty((54,3))
# FaceCamera（摄像头朝向，摄像头编号，控制按键，视角）
# camera = FaceCamera('U', U_index,'q',1)
# all_colors[0:9] = camera.process()  # U
#
# # 夹爪执行左爪D到B的调整，即a）
# tran = a()
# for z in tran:
#     z = sorts(z)
#     print(z)
#     ser.write(z.encode())  # 将调整过程写入arduino
#     while(1):
#         trans = ser.readline()
#         if trans != b'':
#             trans = trans.split()[0].decode()
#             if trans == z + "next":
#                 break
#     print(trans)
#     time.sleep(1)
#
#
# camera = FaceCamera('F', F_index,'q',0)
# all_colors[18:27] = camera.process()  # F
# # 夹爪执行B到U的调整,a)
# tran = a()
# for z in tran:
#     z = sorts(z)
#     print(z)
#     ser.write(z.encode())  # 将调整过程写入arduino
#     while (1):
#         trans = ser.readline()
#         if trans != b'':
#             trans = trans.split()[0].decode()
#             if trans == z + "next":
#                 break
#     print(trans)
#     time.sleep(1)
#
# camera = FaceCamera('D', F_index,'q',1)  # 和F是同一个摄像头
# all_colors[27:36] = camera.process()  # D
# # 夹爪执行U到D的调整，即d)
# tran = d()
# for z in tran:
#     z = sorts(z)
#     print(z)
#     ser.write(z.encode())  # 将调整过程写入arduino
#     while (1):
#         trans = ser.readline()
#         if trans != b'':
#             trans = trans.split()[0].decode()
#             if trans == z + "next":
#                 break
#     print(trans)
#     time.sleep(1)
#
# # 换左摄像头
# camera = FaceCamera('L', L_index,'q',0)
# all_colors[36:45] = camera.process()  # L
# # 夹爪执行右爪R到F的调整，即e)
# tran = ee()
# for z in tran:
#     z = sorts(z)
#     print(z)
#     ser.write(z.encode())  # 将调整过程写入arduino
#     while (1):
#         trans = ser.readline()
#         if trans != b'':
#             trans = trans.split()[0].decode()
#             if trans == z + "next":
#                 break
#     print(trans)
#     time.sleep(1)
#
# camera = FaceCamera('B', B_index,'q',0)
# all_colors[45:54] = camera.process()  # B
# # 夹爪执行F到L的调整，e)
# tran = ee()
# for z in tran:
#     z = sorts(z)
#     print(z)
#     ser.write(z.encode())  # 将调整过程写入arduino
#     while (1):
#
#         trans = ser.readline()
#         if trans != b'':
#             trans = trans.split()[0].decode()
#             if trans == z + "next":
#                 break
#     print(trans)
#     time.sleep(1)
#
# camera = FaceCamera('R', B_index,'q',0)  # 和LB是同一个摄像头
# all_colors[9:18] = camera.process()  # R
# # 夹爪执行L到R的调整,h）
# tran = h()
# for z in tran:
#     z = sorts(z)
#     print(z)
#     ser.write(z.encode())  # 将调整过程写入arduino
#     while (1):
#         trans = ser.readline()
#         if trans != b'':
#             trans = trans.split()[0].decode()
#             if trans == z + "next":
#                 break
#     print(trans)
#     time.sleep(2)
# print(all_colors)

# # 聚类为 6 类
# NUM_CLUSTERS = 6
# kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25, avoid_empty_clusters=True)
# assigned_clusters = kclusterer.cluster(all_colors, assign_clusters=True)
# print('聚类结果\n', assigned_clusters)
# cube_position = assigned_clusters

imageU = cv2.imread('temp_camera/camera_frame_oriU.png')
imageU = cv2.transpose(imageU)
imageU = cv2.flip(imageU, 0)
num,all_colors[0:9] = findColors(imageU)
imageF = cv2.imread('temp_camera/camera_frame_oriF.png')
imageF = cv2.transpose(imageF)
imageF = cv2.flip(imageF, 0)
num,all_colors[18:27] = findColors(imageF)
imageD = cv2.imread('temp_camera/camera_frame_oriD.png')
imageD = cv2.transpose(imageD)
imageD = cv2.flip(imageD, 0)
num,all_colors[27:36] = findColors(imageD)
imageL = cv2.imread('temp_camera/camera_frame_oriL.png')
num,all_colors[36:45] = findColors(imageL)
imageB = cv2.imread('temp_camera/camera_frame_oriB.png')
num,all_colors[45:54] = findColors(imageB)
imageR = cv2.imread('temp_camera/camera_frame_oriR.png')
num,all_colors[9:18] = findColors(imageR)

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
    result = processResults(results)
    print('合作型机械夹爪运行公式：',result)
    turn = ' '.join(result)
    print(turn)


else:
    print('魔方无效')

while(1):
    command = input()
    if command == 'z':
        break

# 发送至 Arduino
# while (1):
#     for z in result:
#         z = sorts(z)
#         print(z)
#         ser.write(z.encode())  # 将调整过程写入arduino
#         while (1):
#
#             trans = ser.readline()
#             if trans != b'':
#                 trans = trans.split()[0].decode()
#                 if trans == z + "next":
#                     break
#             time.sleep(1)
#         print(trans)
#
#     ser.write('x'.encode())
#     trans = ser.readline()
#     if trans != b'':
#         trans = trans.split()[0].decode()
#         print(trans)
#         if trans == "Stop":
#             print("Already finish!")
#             break
#
#     time.sleep(2)