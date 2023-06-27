# from nltk.cluster.kmeans import KMeansClusterer
# import nltk.cluster.util
import numpy as np
import cv2
from utils import *
from solveCube import *
from solve import *
import matplotlib.pyplot as plt
from cameraProcess import *
from k_means import *


# U R F D L B
# 调试用的预设值
# #控制Python中小数的显示精度；suppress：小数是否需要以科学计数法的形式输出
np.set_printoptions(suppress=True)

# 摄像头编号
# 3 1 4 2
F_index = 1
B_index = 1
L_index = 1
U_index = 1

# # 存放六个面共54个色块的rbg像素值
all_colors = np.empty((54,3))
# FaceCamera（摄像头朝向，摄像头编号，控制按键，视角）
# camera = FaceCamera('U', U_index,'q',0)
# all_colors[0:9] = camera.process()  # U
#
# camera = FaceCamera('F', F_index,'q',0)
# all_colors[18:27] = camera.process()  # F
# # 左：松爪 右：转90度
#
# camera = FaceCamera('D', F_index,'q',0)  # 和F是同一个摄像头
# all_colors[27:36] = camera.process()  # D
#
# camera = FaceCamera('L', L_index,'q',0)
# all_colors[36:45] = camera.process()  # L
#
# camera = FaceCamera('B', B_index,'q',0)
# all_colors[45:54] = camera.process()  # B
#
# camera = FaceCamera('R', F_index,'q',0)  # 和F是同一个摄像头
# all_colors[9:18] = camera.process()  # R
# 此时F和B只拍到8个色块q

# print(all_colors)

# # 聚类为 6 类
# NUM_CLUSTERS = 6
# kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25, avoid_empty_clusters=True)
# assigned_clusters = kclusterer.cluster(all_colors, assign_clusters=True)
# print('聚类结果\n', assigned_clusters)
# cube_position = assigned_clusters

imageU = cv2.imread('temp_camera/camera_frame_oriU.png')
all_colors[0:9] = findColors(imageU)
imageF = cv2.imread('temp_camera/camera_frame_oriF.png')
all_colors[18:27] = findColors(imageF)
imageD = cv2.imread('temp_camera/camera_frame_oriD.png')
all_colors[27:36] = findColors(imageD)
imageL = cv2.imread('temp_camera/camera_frame_oriL.png')
all_colors[36:45] = findColors(imageL)
imageB = cv2.imread('temp_camera/camera_frame_oriB.png')
all_colors[45:54] = findColors(imageB)
imageR = cv2.imread('temp_camera/camera_frame_oriR.png')
all_colors[9:18] = findColors(imageR)

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
    # result = processResults(results)
    result = processResults2(results)
    print('合作型机械夹爪运行公式：',result)
    turn = ' '.join(result)
    print(turn)
    # 发送至 Arduino

else:
    print('魔方无效')








