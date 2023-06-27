import numpy as np
import cv2
import math
from collections import Counter


# 返回分类后的可视化色块
def showColor(face_index, face_colors, face_cls):

    face_colors_img = np.empty((300, 300, 3), dtype='uint8')

    for i, values in enumerate(face_colors):
        # 绘制此色块的100*100图像
        image = np.empty((3, 100, 100), dtype='uint8')
        for j in range(3):
            image[j, :, :] = values[j]
        image = image.transpose(2, 1, 0)

        # 将分类结果写入图片
        img_name = 'temp_colors/Face{}_Color{}.png'.format(face_index + 1, i + 1)
        # 如果不先存储再读取，直接写入文本会报错。前后类型都是array。未解之谜
        cv2.imwrite(img_name, image)
        image = cv2.imread(img_name)
        cls = face_cls[i]
        #cv2.putText(image, str(cls), (40, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 2)

        # 保存最终图片
        cv2.imwrite(img_name, image)
        # cv2.imshow('frame_show_{}'.format(i),image)

        # i+1 从 1 到 9
        index_x = math.ceil((i+1)/3)
        index_y = (i+1)%3
        if index_y == 0:
            index_y = 3
        face_colors_img[(index_x-1)*100:index_x*100, (index_y-1)*100:index_y*100, :] = image

    # 将此面九个色块拼接展示
    cv2.imwrite('temp_colors/AFace{}.png'.format(face_index+1), face_colors_img)


def cubeIsValid(cube_position, center_cls):
    # 中心点类别不重复
    set_lst = set(center_cls)
    if len(set_lst) is not len(center_cls):
        print('无法识别出6个中心色块')
        return False
    # 检查每种色块个数
    counts = Counter(cube_position)
    for i in range(6):
        if counts[i] is not 9:
            print('色块数量有误')
            return False

    return True
