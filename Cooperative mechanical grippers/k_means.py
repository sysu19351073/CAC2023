from numpy import *


def distance_euclidean(vector1, vector2):
    """计算欧氏距离"""
    return sqrt(sum(power(vector1-vector2, 2)))  # 返回两个向量的距离


def k_means(all_colors, centers_color, distMeas=distance_euclidean):
    """K-means聚类算法"""
    # mat函数将列表存为矩阵
    all_colors = mat(all_colors)
    # 建立簇分配结果矩阵，第一列存放该数据所属中心点，第二列是该数据到中心点的距离
    clusterAssment = mat(zeros((54, 2)))
    # 以六个中心色块作为质心
    centroids = mat(centers_color)
    # 用来判定聚类是否收敛
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(54):  # 把每一个数据划分到离他最近的中心点
            minDist = inf # 无穷大
            minIndex = -1 #初始化
            for j in range(6):
                # 计算各点与新的聚类中心的距离
                distJI = distMeas(centroids[j, :], all_colors[i, :])
                if distJI < minDist:
                    # 如果第i个数据点到第j中心点更近，则将i归属为j
                    minDist = distJI
                    minIndex = j
            # 如果分配发生变化，则需要继续迭代
            if clusterAssment[i,0] != minIndex:
                clusterChanged = True
            # 并将第i个数据点的分配情况存入字典
            clusterAssment[i,:] = minIndex,minDist**2
        # print(centroids)
        for cent in range(6):  # 重新计算中心点
            # 取第一列等于cent的所有列
            ptsInClust = all_colors[nonzero(clusterAssment[:, 0].A == cent)[0]]
            # 算出这些数据的中心点
            centroids[cent, :] = mean(ptsInClust, axis=0)
    clusterAssment_list = []
    for i in range(54):
        clusterAssment_list.append(int(clusterAssment[i,0]))
    return clusterAssment_list




