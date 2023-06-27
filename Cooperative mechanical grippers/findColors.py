import cv2
import math
import numpy as np
from utils import *
import time


# 调参，进程膨胀处理
def operate(img):
    # Process a bgr image to binary
    # kernel = np.ones((3,3),np.uint8) this is an alternative way to create kernel
    # 改变图像的颜色空间，该函数形式为：
    # cv2.cvtColor(frame,cv2.COLOR_BGR2RGB);
    # frame为要进行处理的图片;cv2.COLOR_BGR2RGB为要进行的色彩转换方式
    # cv2.COLOR_BGR2GRAY转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Corresponding grayscale image to the input
    # cv2.adaptiveThreshold是一个用于二值化图像的函数，使用自适应阈值来转换图像
    # adaptiveThreshold(src,dst,maxValue,adaptiveMethod,thresholdType,blockSize, C);
    # （输入图像，阈值最大值，自适应阈值算法类型，阈值操作的类型，窗口的大小-只能为奇数，自适应阈值算法得到平均值或加权平均值后再减的常数值）
    # 自适应阈值算法类型：0为ADAPTIVE_THRESH_MEAN_C（均值法获取阈值），1为ADAPTIVE_THRESH_GAUSSIAN_C（高斯窗加权和获取阈值）
    # 1为THRESH_BINARY_INV（反向二值化）
    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    # cv2.getStructuringElement(shape， size) 构造形态学的核
    # shape：代表形状类型-cv2. MORPH_ELLIPSE：椭圆形结构元素；size：代表形状元素的大小3x3
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # cv2.medianBlur（src,k-size）中值滤波函数，k-size为滤波核的大小，为比1大的奇数
    binary_blurred = cv2.medianBlur(binary, 7)
    # binary_erosion = cv2.erode(binary_blurred, kernel, iterations=1)
    # binary_blurred = binary_erosion
    # cv2.dilate(img, kernel, iteration)，kernel即膨胀操作的内核，iteration为膨胀次数
    binary_dilated = cv2.dilate(binary_blurred, kernel, iterations=3)  # 将膨胀操作进行很多次
    # 发现蓝色色块的轮廓会出现断裂，尝试调节kernel大小和迭代次数
    # # 增加腐蚀操作，此操作后来没有用
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4))
    # binary_erosion = cv2.erode(binary_dilated, kernel, iterations=4)
    # binary_blurred = binary_erosion
    # cv2.imshow('binary_erosion', binary_erosion)
    # cv2.imshow('binary', binary)
    # cv2.imshow('binary_dilated',binary_dilated)
    import time
    cv2.imwrite('temp_camera_colors/{}.jpg'.format(time.time()), binary_dilated)

    binary_inv = 255 - binary_dilated

    return binary_inv


# 调参，进行矩阵筛选
def approx_is_square(approx, SIDE_VS_SIDE_THRESHOLD=0.70, ANGLE_THRESHOLD=30, ROTATE_THRESHOLD=30):
    """
    Rules
    - there must be four corners
    - all four lines must be roughly the same length
    - all four corners must be roughly 90 degrees
    - AB and CD must be horizontal lines
    - AC and BC must be vertical lines
    SIDE_VS_SIDE_THRESHOLD
        If this is 1 then all 4 sides must be the exact same length.  If it is
        less than one that all sides must be within the percentage length of
        the longest side.
        A ---- B
        |      |
        |      |
        C ---- D
    """

    # assert断言，判断语句是否为FALSE，若是，则触发异常
    assert SIDE_VS_SIDE_THRESHOLD >= 0 and SIDE_VS_SIDE_THRESHOLD <= 1, "SIDE_VS_SIDE_THRESHOLD must be between 0 and 1"
    assert ANGLE_THRESHOLD >= 0 and ANGLE_THRESHOLD <= 90, "ANGLE_THRESHOLD must be between 0 and 90"

    # There must be four corners[]
    if len(approx) != 4:
        return False

    # Find the four corners
    (A, B, C, D) = sort_corners(tuple(approx[0][0]),
                                tuple(approx[1][0]),
                                tuple(approx[2][0]),
                                tuple(approx[3][0]))

    # Find the lengths of all four sides
    AB = pixel_distance(A, B)
    AC = pixel_distance(A, C)
    DB = pixel_distance(D, B)
    DC = pixel_distance(D, C)
    distances = (AB, AC, DB, DC)
    max_distance = max(distances)
    cutoff = int(max_distance * SIDE_VS_SIDE_THRESHOLD)
    # print(distances)
    # 除了控制矩形的比例，还可以控制长度
    min_length = 40
    max_length = 200

    # If any side is much smaller than the longest side, return False
    for distance in distances:
        if distance < cutoff:
            return False
        elif distance > max_length or distance < min_length:
            return False

    return True


def sort_corners(corner1, corner2, corner3, corner4):
    """
    Sort the corners such that
    - A is top left
    - B is top right
    - C is bottom left
    - D is bottom right
    Return an (A, B, C, D) tuple
    """
    results = []
    corners = (corner1, corner2, corner3, corner4)

    min_x = None
    max_x = None
    min_y = None
    max_y = None

    for (x, y) in corners:
        if min_x is None or x < min_x:
            min_x = x

        if max_x is None or x > max_x:
            max_x = x

        if min_y is None or y < min_y:
            min_y = y

        if max_y is None or y > max_y:
            max_y = y

    # top left
    top_left = None
    top_left_distance = None
    for (x, y) in corners:
        # 像素距离
        distance = pixel_distance((min_x, min_y), (x, y))
        if top_left_distance is None or distance < top_left_distance:
            top_left = (x, y)
            top_left_distance = distance

    results.append(top_left)

    # top right
    top_right = None
    top_right_distance = None

    for (x, y) in corners:
        if (x, y) in results:
            continue

        distance = pixel_distance((max_x, min_y), (x, y))
        if top_right_distance is None or distance < top_right_distance:
            top_right = (x, y)
            top_right_distance = distance
    results.append(top_right)

    # bottom left
    bottom_left = None
    bottom_left_distance = None

    for (x, y) in corners:
        if (x, y) in results:
            continue

        distance = pixel_distance((min_x, max_y), (x, y))

        if bottom_left_distance is None or distance < bottom_left_distance:
            bottom_left = (x, y)
            bottom_left_distance = distance
    results.append(bottom_left)

    # bottom right
    bottom_right = None
    bottom_right_distance = None

    for (x, y) in corners:
        if (x, y) in results:
            continue

        distance = pixel_distance((max_x, max_y), (x, y))

        if bottom_right_distance is None or distance < bottom_right_distance:
            bottom_right = (x, y)
            bottom_right_distance = distance
    results.append(bottom_right)

    return results


def pixel_distance(A, B):
    """
    Pythagrian therom to find the distance between two pixels
    """
    (col_A, row_A) = A
    (col_B, row_B) = B

    return (math.sqrt(math.pow(col_B - col_A, 2) + math.pow(row_B - row_A, 2)) + 0)


def sort_x(n,k,siu):
    """
    sort z first
    """
    siuu = []
    for i in range(n):
        for j in range(n):
            if siu[i+k][0][0] <= siu[j+k][0][0]:
                siuu = siu[i+k]
                siu[i+k] = siu[j+k]
                siu[j+k] = siuu
    return siu


def sort_y(n,k,siu):
    """
    then sort y by three
    """
    siuu = []
    for i in range(n):
        for j in range(n):
            if siu[i+k][0][1] <= siu[j+k][0][1]:
                siuu = siu[i+k]
                siu[i+k] = siu[j+k]
                siu[j+k] = siuu
    return siu


def sort_cube(siu, n):
    """
    sort cube base on (z,y)
    """
    if n == 9:
        siu = sort_y(n, 0, siu)
        siu = sort_x(3, 0, siu)
        siu = sort_x(3, 3, siu)
        siu = sort_x(3, 6, siu)
    else:
        siu = siu

    return siu


def findColors(image):
    # 将 3 通道 BGR 彩色图像分离为 B、G、R 单通道图像
    # b, g, r = cv2.split(image)
    img = image.copy()
    recnum = 0
    # row*col二维列表创建 list[[0,0]...9[0,0]]
    cords = [[0 for col in range(2)] for row in range(9)]
    # 得到处理后的图像（灰度、二值化、滤波、膨胀）
    dilation = operate(image)
    # contours,hierarchy = cv2.findContours(contour,TREE,SIMPLE)轮廓提取
    # （带有轮廓信息的图像；输出轮廓中只有外侧轮廓信息；压缩水平方向、垂直方向、对角线方向的元素，只保留该方向的终点坐标）
    # contours：list结构，列表中每个元素代表一个边沿信息；每个元素是(z,1,2)的三维向量，x表示该条边沿里共有多少个像素点，第三维的那个“2”表示每个点的横、纵坐标
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    approves = []
    for cnt in contours:
        # 猜测魔方的圆角会影响cv2.approxPolyDP
        # cv2.approxPolyDP(curve,epsilon,closed)轮廓线多边形逼近
        # curve：轮廓点的集合；epsilon：指定近似精度的参数，原始曲线和它的近似之间最大距离；closed：如果为true，则闭合近似曲线，否则，不闭合
        approx = cv2.approxPolyDP(cnt, 0.1 * cv2.arcLength(cnt, True), True)

        # ravel数组降维
        x = approx.ravel()[0]
        y = approx.ravel()[1]
        # if (len(approx) == 4 and 0 < z < 1000 and 0 < y < 1000):
        # 在这里限定了魔方的位置
        if len(approx) == 4 and 0 < x < 1000 and 0 < y < 1000:
            # 形状为矩形
            if approx_is_square(approx):
                # 计算
                # Approx has 4 (z,y) coordinates, where the first is the top left,and
                # the third is the bottom right. Findind the mid point of these two coordinates
                # will give me the center of the rectangle
                approves.append(approx)
                recnum = recnum + 1
                x1 = approx[0, 0, 0]
                y1 = approx[0, 0, 1]
                x2 = approx[(approx.shape[0] - 2), 0, 0]  # X coordinate of the bottom right corner
                y2 = approx[(approx.shape[0] - 2), 0, 1]

                xavg = int((x1 + x2) / 2)
                yavg = int((y1 + y2) / 2)

                if (recnum > 9):
                    break
                cords = list(cords)
                cords[recnum - 1] = [xavg, yavg]
    index = 0
    siu = []

    if len(approves) != 0:
        if len(cords) != 0:
            if len(approves) == len(cords):
            # 合并中心点和轮廓线
                for approx in approves:
                    siu.append([cords[index], approx])
                    index = index + 1
    n = index

    # 色块排序
    siu = sort_cube(siu, n)
    # print(siu)

    cords = []
    approves = []
    for siuu in siu:
        cords.append(siuu[0])
        approves.append(siuu[1])
        index = index + 1

    # 得到九个中心点坐标cords，计算face_colors
    rec_size = 25
    face_colors = np.empty((9, 3))
    target = 0
    num = 0
    for i, cord in enumerate(cords):
        target = image[cord[1] - rec_size:cord[1] + rec_size, cord[0] - rec_size:cord[0] + rec_size]
        # 注意这里xy反过来，因为image形状是高*宽
        # print(z-rec_size,z+rec_size,y-rec_size,y+rec_size)
        print('色块{}：中心点{} {}'.format(i+1, cord[0], cord[1]))
        # cv2.imshow('frame_show_{}'.format(i), target)
        # 处理色块：求框选区域的均值
        b = np.mean(target[:, :, 0])
        g = np.mean(target[:, :, 1])
        r = np.mean(target[:, :, 2])
        target = [b, g, r]

        # target = np.squeeze(np.mean(np.mean(target, axis=0), axis=1))
        face_colors[i, :] = target  # 第 i 个色块
        num = i+1


    index = 0
    for approx in approves:
        # 可视化调试
        cv2.circle(img, cords[index], 15, (255, 255, 255), 5)
        # cv2.putText(image,str(b[yavg,xavg])+str(g[yavg,xavg])+str(r[yavg,xavg]),(100,recnum*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255))
        cv2.drawContours(img, [approx], 0, (0, 0, 255), 2)
        cv2.putText(img, str(index + 1), cords[index], cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
        index = index + 1
    cv2.imwrite('temp_camera_colors/{}.jpg'.format(time.time()), img)
    return num,face_colors  # 得到(9,3)


def findColors2(frame, center_restrain):
    center_frame = frame[center_restrain[0][1]:center_restrain[1][1], center_restrain[0][0]:center_restrain[1][0]]
    cv2.rectangle(frame, (center_restrain[0][0], center_restrain[0][1]),
                  (center_restrain[1][0], center_restrain[1][1]),
                  (150, 150, 150), 3)

    dilation = operate(center_frame)

    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # 猜测魔方的圆角会影响cv2.approxPolyDP
        approx = cv2.approxPolyDP(cnt, 0.12 * cv2.arcLength(cnt, True), True)
        x = approx.ravel()[0]
        y = approx.ravel()[1]

        if (len(approx) == 4 and 0 < x < 1000 and 0 < y < 1000):
            # 形状为矩形
            if (approx_is_square(approx) == True):
                print(approx.shape)
                x1 = approx[0, 0, 0]
                y1 = approx[0, 0, 1]
                x2 = approx[(approx.shape[0] - 2), 0, 0]  # X coordinate of the bottom right corner
                y2 = approx[(approx.shape[0] - 2), 0, 1]

                break

    # 获得中心点的矩形拟合坐标approx (4,1,2)

    # 将截取图上的坐标映射到原图
    for i in range(4):
        approx[i][0][0] = approx[i][0][0] + center_restrain[0][0]
        approx[i][0][1] = approx[i][0][1] + center_restrain[0][1]

    # 可视化调试
    cv2.drawContours(image, [approx], 0, (0, 0, 255), 2)
    cv2.imshow('ff', image)
    cv2.waitKey(0)

    # 根据中心色块拟合的矩形，预测整个魔方面的形状


if __name__ == '__main__':
    image = cv2.imread('temp_camera/U.png')
    # center_restrain = [[230,160],[450,350]]
    face_index = 0
    face_color = findColors(image)
    print(len(face_color))
    face_cls = []
    showColor(face_index, face_color, face_cls)
    time.sleep(10)
    # findColors2(image, center_restrain)
