import kociemba
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def cubeSolve(cube_position, centers_cls):
    # 替换
    cube_position = ['U' if i == centers_cls[0] else i for i in cube_position]
    cube_position = ['R' if i == centers_cls[1] else i for i in cube_position]
    cube_position = ['F' if i == centers_cls[2] else i for i in cube_position]
    cube_position = ['D' if i == centers_cls[3] else i for i in cube_position]
    cube_position = ['L' if i == centers_cls[4] else i for i in cube_position]
    cube_position = ['B' if i == centers_cls[5] else i for i in cube_position]
    cube_position = "".join(cube_position)  # 连为一个string
    print('魔方姿态：', cube_position)
    results = kociemba.solve(cube_position)
    return results

# 解形式转化
def coverlist(cube_position):
    # 将 kociemba 解处理为需要的形式
    trans = cube_position.split()
    formula = []
    formula_rot = []
    for z in range(len(trans)):
        if z == 0:
            pretran = "D"
        else:
            pretran = list(trans[z - 1])[0]

        if len(trans[z]) == 1:
            tran1 = trans[z]
            tran2 = "0"
        if len(trans[z]) > 1:
            tran1 = list(trans[z])[0]
            tran2 = list(trans[z])[1]

        formula.append(tran1)
        formula_rot.append(tran2)

    # cube_position = " ".join(cube_position)  # 连为一个string
    return formula, formula_rot

# 状态字符转为数字
def tran_to_Num(formula):
    # 将相对的面定义为相反数
    # F\B = 1\-1;U\D = 2\-2;R\L = 3\-3
    Numformula = []
    for tran in formula:
        if tran == "F":
            tran = 1
            Numformula.append(tran)
            continue
        if tran == "B":
            tran = -1
            Numformula.append(tran)
            continue
        if tran == "U":
            tran = 2
            Numformula.append(tran)
            continue
        if tran == "D":
            tran = -2
            Numformula.append(tran)
            continue
        if tran == "R":
            tran = 3
            Numformula.append(tran)
            continue
        if tran == "L":
            tran = -3
            Numformula.append(tran)
            continue
    return Numformula

# 结点类
class Node(object):
    # 各结点存储的数据格式为
    # node(左爪夹的面，右爪夹的面，属于第几步的结点，指向下一步的路径和cost，编号）
    # state为夹持魔方的状态，格式为[x,y]
    # path为路径和距离，格式为{(start,final):cost}
    def __init__(self,state,Step,path,No):
        self.state = state
        self.Step = Step
        self.path = path
        self.No = No

    def path_append(self,Newpath):
        self.path.update(Newpath)

    def printNode(self):
        #To_print.append(self.state)
        print(self.path)

# 图构建函数
def Graph(formula):
    # 构建魔方复原过程图
    # x = "D";y = "R"
    x = -2
    y = 3
    state = [x, y]# 初始状态
    i = 0;j = 0 #i为每一层的序号，j为每个节点的序号
    nodes = [[]]
    graph = {}
    path_cost = {}
    node = Node(state,i,path_cost,j)
    nodes[0].append(node)
    # 复原公式中的每一步视为一个阶段
    for i in range(len(formula)):
        graph[i] = nodes[i]
        nodes.append([])
        if nodes[i + 1]:
            pass
        else:
            nodes.append([])
        # 该阶段的结点由上一阶段中的结点生成
        for node in nodes[i]:
            # 初始化路径字典
            path_cost = {};path_cost0 = {};path_cost1 = {};path_cost2 = {}
            j = j + 1
            state = node.state
            exist = 0 # 定义状态指示器指示状态是否一致
            #print(node.path)
            ## 两步前进
            # 若最后不能走两步则不进入循环
            if i + 2 <= len(formula):
                while(True):
                    # 若需进行复原的两面相对，则退出循环
                    if formula[i] + formula[i+1] == 0:
                        break
                    else:
                        path_cost00 = {}
                        path_cost01 = {}
                        path_cost02 = {}
                        # 若夹爪夹的一面与需复原的两面中的一面相对
                        # 左
                        if state[0] + formula[i] == 0 or state[0] + formula[i + 1] == 0:
                            if state[0] + formula[i] == 0:
                                state0 = [formula[i], formula[i + 1]]
                                newnode0 = Node(state0, i + 1, path_cost00, j)
                                for exnode in nodes[i + 2]:
                                    if newnode0.state == exnode.state:
                                        newnode0.No = exnode.No
                                        j = j - 1
                                        exist = 1
                                        break
                                if exist == 0:
                                    nodes[i + 2].append(newnode0)
                                else:
                                    exist = 0
                                # nodes[i + 2].append(newnode0)
                                path = (node.No, newnode0.No)
                                cost = 10
                                path_cost[path] = cost
                                node.path_append(path_cost)
                                j = j + 1
                                break
                            elif state[0] + formula[i + 1] == 0:
                                state0 = [formula[i + 1], formula[i]]
                                newnode0 = Node(state0, i + 1, path_cost00, j)
                                for exnode in nodes[i + 2]:
                                    if newnode0.state == exnode.state:
                                        newnode0.No = exnode.No
                                        j = j - 1
                                        exist = 1
                                        break
                                if exist == 0:
                                    nodes[i + 2].append(newnode0)
                                else:
                                    exist = 0
                                # nodes[i + 2].append(newnode0)
                                path = (node.No, newnode0.No)
                                cost = 10
                                path_cost[path] = cost
                                node.path_append(path_cost)
                                j = j + 1
                                break
                        # 右
                        elif state[1] + formula[i] == 0 or state[1] + formula[i + 1] == 0:
                            if state[1] + formula[i] == 0:
                                state0 = [formula[i + 1], formula[i]]
                                newnode0 = Node(state0, i + 1, path_cost00, j)
                                for exnode in nodes[i + 2]:
                                    if newnode0.state == exnode.state:
                                        newnode0.No = exnode.No
                                        j = j - 1
                                        exist = 1
                                        break
                                if exist == 0:
                                    nodes[i + 2].append(newnode0)
                                else:
                                    exist = 0
                                # nodes[i + 2].append(newnode0)
                                path = (node.No, newnode0.No)
                                cost = 10
                                path_cost[path] = cost
                                node.path_append(path_cost)
                                j = j + 1
                                break
                            elif state[1] + formula[i + 1] == 0:
                                state0 = [formula[i], formula[i + 1]]
                                newnode0 = Node(state0, i + 1, path_cost00, j)
                                for exnode in nodes[i + 2]:
                                    if newnode0.state == exnode.state:
                                        newnode0.No = exnode.No
                                        j = j - 1
                                        exist = 1
                                        break
                                if exist == 0:
                                    nodes[i + 2].append(newnode0)
                                else:
                                    exist = 0
                                # nodes[i + 2].append(newnode0)
                                path = (node.No, newnode0.No)
                                cost = 10
                                path_cost[path] = cost
                                node.path_append(path_cost)
                                j = j + 1
                                break

                        # 若夹爪夹的一面与需复原的两面中的一面相同，包含在单步中
                        # 下一面即可直接复原
                        if formula[i] == state[0] or formula[i] == state[1]:
                            break
                        elif formula[i + 1] == state[0] or formula[i + 1] == state[1]:
                            break

                        # 若需复原的两面有操作空间
                        state01 = [formula[i], formula[i + 1]]
                        state02 = [formula[i + 1], formula[i]]
                        newnode01 = Node(state01, i + 1, path_cost01, j)
                        for exnode in nodes[i + 2]:
                            if newnode01.state == exnode.state:
                                newnode01.No = exnode.No
                                j = j - 1
                                exist = 1
                                break
                        if exist == 0:
                            nodes[i + 2].append(newnode01)
                        else:
                            exist = 0
                        j = j + 1
                        newnode02 = Node(state02, i + 1, path_cost02, j)
                        j = j + 1
                        for exnode in nodes[i + 2]:
                            if newnode02.state == exnode.state:
                                newnode02.No = exnode.No
                                j = j - 1
                                exist = 1
                                break
                        if exist == 0:
                            nodes[i + 2].append(newnode02)
                        else:
                            exist = 0
                        # nodes[i + 2].append(newnode01)
                        # nodes[i + 2].append(newnode02)

                        path = (node.No, newnode01.No)
                        cost = 10
                        path_cost[path] = cost
                        node.path_append(path_cost)

                        path = (node.No, newnode02.No)
                        cost = 10
                        path_cost[path] = cost
                        node.path_append(path_cost)

                        break


            ## 单步前进
            # 待复原的面被包含在状态中
            if formula[i] in state:
                newnode = Node(state, i, path_cost0, j)# 创建新的结点
                # 合并状态一样的结点
                for exnode in nodes[i + 1]:
                    if newnode.state == exnode.state:
                        newnode.No = exnode.No
                        j = j - 1
                        exist = 1
                        break
                if exist == 0:
                    nodes[i + 1].append(newnode)
                else:
                    exist = 0
                path = (node.No, newnode.No)
                cost = 0
                path_cost[path]=cost
                node.path_append(path_cost)# 更新该结点的路径字典
                continue
            state1 = [state[0], formula[i]]
            state2 = [formula[i], state[1]]

            # 待复原的面为其中一个面的相对面
            if formula[i]+state[0] == 0:
                newnode2 = Node(state2, i, path_cost2, j)# 创建新的结点
                for exnode in nodes[i + 1]:
                    if newnode2.state == exnode.state:
                        newnode2.No = exnode.No
                        j = j - 1
                        exist = 1
                        break
                if exist == 0:
                    nodes[i + 1].append(newnode2)
                else:
                    exist = 0
                state = state2
                path = (node.No, newnode2.No)
                cost = 6
                path_cost[path] = cost
                node.path_append(path_cost)
                continue
            elif formula[i]+state[1] == 0:
                newnode1 = Node(state1, i, path_cost1, j)# 创建新的结点
                for exnode in nodes[i + 1]:
                    if newnode1.state == exnode.state:
                        newnode1.No = exnode.No
                        j = j - 1
                        exist = 1
                        break
                if exist == 0:
                    nodes[i + 1].append(newnode1)
                else:
                    exist = 0
                state = state1
                path = (node.No, newnode1.No)
                cost = 6
                path_cost[path] = cost
                node.path_append(path_cost)
                continue

            newnode1 = Node(state1, i, path_cost1, j)
            for exnode in nodes[i + 1]:
                if newnode1.state == exnode.state:
                    newnode1.No = exnode.No
                    j = j - 1
                    exist = 1
                    break
            if exist == 0:
                nodes[i + 1].append(newnode1)
            else:
                exist = 0

            j = j + 1

            newnode2 = Node(state2, i, path_cost2, j)
            for exnode in nodes[i + 1]:
                if newnode2.state == exnode.state:
                    newnode2.No = exnode.No
                    j = j - 1
                    exist = 1
                    break
            if exist == 0:
                nodes[i + 1].append(newnode2)
            else:
                exist = 0

            path = (node.No, newnode1.No)
            cost = 6
            path_cost[path] = cost
            node.path_append(path_cost)

            path = (node.No, newnode2.No)
            cost = 6
            path_cost[path] = cost
            node.path_append(path_cost)


    graph[i+1] = nodes[i+1]

    Gra = {}
    end = []
    point = []
    path_real = {}
    for i in range(len(formula)+1):
        for node in nodes[i]:
            if node.path == {}:
                end.append(node.No)

            for key in node.path:
                no1 = str(key[0])
                no2 = str(key[1])
                key_real = (no1, no2)

                path_real[key_real] = node.path[key]
            Gra.update(node.path)
            point.append(node.No)
            #print(node.No)
            #print(node.state)
            #print(node.path)
    point = set(point)
    # print(point)
    # print(end)
    # print(Gra)
    return graph, Gra, point, end

# 绘图函数
def pic_graph(Nodes, Arcs):
    """
    可视化图，节点的相对位置可能改变
    """
    np.random.seed(1)
    # 定义有向图
    Graph = nx.DiGraph()
    # 定义节点
    for node in Nodes:
        Graph.add_node(node, min_dis=0, previous_node=None, node_name=node)
    # 定义弧
    for (from_node, to_node), weight in Arcs.items():
        Graph.add_edge(from_node, to_node, weight=weight)

    plt.figure(figsize=(7, 5), dpi=100)
    pos = nx.spring_layout(Graph)

    nx.draw_networkx(Graph, pos)
    node_labels = nx.get_node_attributes(Graph, 'node_name')
    nx.draw_networkx_labels(Graph, pos, labels=node_labels)
    edge_labels = nx.get_edge_attributes(Graph, 'weight')
    nx.draw_networkx_edge_labels(Graph, pos, edge_labels=edge_labels)

    plt.savefig('./spp_grap.png')
    plt.show()

    return True

# Dijkstra算法
def Dijkstra(start, target, graph):
    if start not in graph['V'] or target not in graph['V']:
        return '无效起点或终点！'

    visited = {}  # 记录顶点的访问情况
    dist = {}  # 记录最短距离
    path = [start]  # 记录最短路径上的每一个顶点
    V, E = graph['V'], graph['E']
    for node in V:
        visited[node] = 0  # 0表示没有访问 1表示已访问
        dist[node] = -1  # -1表示无限远
    dist[start] = 0  # 到自己的距离为0
    visited[start] = 1

    # 初始化到各顶点的距离
    for e in E:
        if e[0] == start:
            dist[e[1]] = E[e]


    # while not all(visited.values()):  # 直到访问完所有顶点 求到所有顶点的最短距离
    while not visited[target]:  # 仅求到终点的距离
        # 在所有可达的但未访问的顶点中，寻找距离最近的一个
        cur_E_li = [(e, E[e]) for e in E if visited[e[0]] and not visited[e[1]]]  # 所有可达但未访问的顶点
        cur_E_li.sort(key=lambda a: a[1])  # 按照距离进行排序
        # print(cur_E_li)
        min_edge_w = cur_E_li[0]  # 第一个就是距离最短的那一个
        new_start_node = min_edge_w[0][1]
        # print(dist)

        # 刷新最短距离
        for node in dist:
            new_edge = (new_start_node, node)
            if new_edge in E:
                if dist[node] == -1:
                    dist[node] = dist[new_start_node] + E[new_edge]
                else:
                    dist[node] = min(dist[node], dist[new_start_node] + E[new_edge])
                path.append(new_start_node)

        # 更新visited
        visited[new_start_node] = 1
        path.append(new_start_node)  # 将当前节点更新到最短路径上


    return path, dist[target]

def Dijkstra_EX(Nodes, Arcs, source, target):

    # 定义有向图
    Graph = nx.DiGraph()
    # 定义节点
    for node in Nodes:
        Graph.add_node(node, min_dis=0, previous_node=None, node_name=node)
    # 定义弧
    for (from_node, to_node), weight in Arcs.items():
        Graph.add_edge(from_node, to_node, weight=weight)

    # ========== 算法开始 ==========
    # 初始化未探索节点集合: 初始时为所有节点集合
    tmp_set = list(Graph.nodes())

    # 初始化当前节点及所有节点的最短距离：起始点作为当前节点，并将起始点的最短路径置为0，其他节点距离置为无穷大
    current_node = source
    for node in tmp_set:
        if (node == source):
            Graph.nodes[node]['min_dis'] = 0
        else:
            Graph.nodes[node]['min_dis'] = np.inf

    # 算法终止条件: 所有节点均被探索
    while (len(tmp_set) > 0):
        # 选择未探索的节点中的最短路径的节点作为新的当前节点
        min_dis = np.inf
        for node in tmp_set:
            if (Graph.nodes[node]['min_dis'] < min_dis):
                current_node = node
                min_dis = Graph.nodes[node]['min_dis']

        # 删除已探索的节点
        if (current_node != None):
            tmp_set.remove(current_node)

        """
        循环判断当前节点所有相邻的节点：
        1. 计算当前节点的最小值 + 相邻节点的权重的和，
        2. 若小于相邻节点的最小距离，则更新相邻节点的最小距离，并标记相邻节点的前一个节点为当前节点
        """
        for neighbor_node in Graph.successors(current_node):
            arc = (current_node, neighbor_node)
            dis_t = Graph.nodes[current_node]['min_dis'] + Graph.edges[arc]['weight']
            if (dis_t < Graph.nodes[neighbor_node]['min_dis']):
                Graph.nodes[neighbor_node]['min_dis'] = dis_t
                Graph.nodes[neighbor_node]['previous_node'] = current_node

    # ========== 获取结果 ==========
    """
    获取结果: 
    1. 结束节点上更新的最短距离即为整个最短路径的距离
    2. 根据结束节点回溯, 依次找到其上一个节点
    """
    distance = Graph.nodes[target]['min_dis']
    current_node = target
    path = [current_node]
    while (current_node != source):
        current_node = Graph.nodes[current_node]['previous_node']
        path.insert(0, current_node)

    # print(f'节点 {source} 到 {target} 的路径={path}, 最短距离={distance}')
    return distance, path


# dijkstra算法内置实现
def networkx_dijkstra(Nodes, Arcs, start, target):
    """
    networkx提供了dijkstra算法的方法的封装，只要定义图结构即可。
    """
    Graph = nx.DiGraph()
    # 添加节点
    for node in Nodes:
        Graph.add_node(node, min_dis=0, previous_node=None, node_name=node)
    # 添加带权重的边
    for (from_node, to_node), weight in Arcs.items():
        Graph.add_weighted_edges_from([(from_node, to_node, weight)])

    path = nx.dijkstra_path(Graph, source=start, target=target)
    distance = nx.dijkstra_path_length(Graph, source=start, target=target)
    # print(f'节点 {start} 到 {target} 的路径={path}, 最短距离={distance}')

    return distance, path


def r_rot(tran):
    ctran = 0
    if tran == "0":
        tran = "R2"
        ctran = "R4"
    if tran == "'":
        tran = "R4"
        ctran = "R2"
    if tran == "2":
        tran = "R3"
        ctran = "R5"
    rot = [tran, "R0", ctran, "R1"]
    return rot

def l_rot(tran):
    ctran = 0
    if tran == "0":
        tran = "L2"
        ctran = "L4"
    if tran == "'":
        tran = "L4"
        ctran = "L2"
    if tran == "2":
        tran = "L3"
        ctran = "L5"
    rot = [tran, "L0", ctran, "L1"]
    return rot

# 将魔方复原公式转为机械夹爪的工作公式
'''
0 : 松开
1 : 夹紧
2 : 顺时针转90
3 : 顺时针转180
4 : 逆时针转90
5 : 逆时针转180
'''

'''
F\B = 1\-1;
U\D = 2\-2;
R\L = 3\-3
'''
def step_tran(path, results, turns):
    term = 0
    hole_tran = []
    for state in path:
        term = term + 1
        if term < len(path):
            next_state = path[term]
        else:
            break
        turn = turns[term - 1]
        next_turn = turns[term]
        result = results[term - 1]
        # 状态相同
        if state == next_state:
            tran = []
            if state[0] == result:
                rot = l_rot(turn)
            else:
                rot = r_rot(turn)

        else:
            # 左状态相同，单步，调整时转动左侧，复原时转动右侧
            if state[0] == next_state[0]:
                # 右相反
                if state[1] + next_state[1] == 0:
                    step = "L3"
                    step_ = "L5"
                else:
                    if state[0] == 1:
                        if (state[1] == 2 and next_state[1] == -3) or (state[1] == -3 and next_state[1] == -2) or (state[1] == -2 and next_state[1] == 3) or (state[1] == 3 and next_state[1] == 2):
                            step = "L2"
                            step_ = "L4"
                        else:
                            step = "L4"
                            step_ = "L2"
                    if state[0] == -1:
                        if (state[1] == 2 and next_state[1] == 3) or (state[1] == 3 and next_state[1] == -2) or (state[1] == -2 and next_state[1] == -3) or (state[1] == -3 and next_state[1] == 2):
                            step = "L2"
                            step_ = "L4"
                        else:
                            step = "L4"
                            step_ = "L2"

                    if state[0] == 2:
                        if (state[1] == 1 and next_state[1] == 3) or (state[1] == 3 and next_state[1] == -1) or (state[1] == -1 and next_state[1] == -3) or (state[1] == -3 and next_state[1] == 1):
                            step = "L2"
                            step_ = "L4"
                        else:
                            step = "L4"
                            step_ = "L2"
                    if state[0] == -2:
                        if (state[1] == 1 and next_state[1] == -3) or (state[1] == -3 and next_state[1] == -1) or (state[1] == -1 and next_state[1] == 3) or (state[1] == 3 and next_state[1] == 1):
                            step = "L2"
                            step_ = "L4"
                        else:
                            step = "L4"
                            step_ = "L2"

                    if state[0] == 3:
                        if (state[1] == 1 and next_state[1] == -2) or (state[1] == -2 and next_state[1] == -1) or (state[1] == -1 and next_state[1] == 2) or (state[1] == 2 and next_state[1] == 1):
                            step = "L2"
                            step_ = "L4"
                        else:
                            step = "L4"
                            step_ = "L2"
                    if state[0] == -3:
                        if (state[1] == 1 and next_state[1] == 2) or (state[1] == 2 and next_state[1] == -1) or (state[1] == -1 and next_state[1] == -2) or (state[1] == -2 and next_state[1] == 1):
                            step = "L2"
                            step_ = "L4"
                        else:
                            step = "L4"
                            step_ = "L2"

                tran = ["R0", step, "R1", "L0", step_, "L1"]
                rot = r_rot(turn)

            # 右状态相同，单步，调整时转动右侧，复原时转动左侧
            elif state[1] == next_state[1]:
                # 左相反
                if state[0] + next_state[0] == 0:
                    step = "R3"
                    step_ = "R5"
                else:
                    if state[1] == 1:
                        if (state[0] == 2 and next_state[0] == -3) or (state[0] == -3 and next_state[0] == -2) or (state[0] == -2 and next_state[0] == 3) or (state[0] == 3 and next_state[0] == 2):
                            step = "R2"
                            step_ = "R4"
                        else:
                            step = "R4"
                            step_ = "R2"
                    if state[1] == -1:
                        if (state[0] == 2 and next_state[0] == 3) or (state[0] == 3 and next_state[0] == -2) or (state[0] == -2 and next_state[0] == -3) or (state[0] == -3 and next_state[0] == 2):
                            step = "R2"
                            step_ = "R4"
                        else:
                            step = "R4"
                            step_ = "R2"

                    if state[1] == 2:
                        if (state[0] == 1 and next_state[0] == 3) or (state[0] == 3 and next_state[0] == -1) or (state[0] == -1 and next_state[0] == -3) or (state[0] == -3 and next_state[0] == 1):
                            step = "R2"
                            step_ = "R4"
                        else:
                            step = "R4"
                            step_ = "R2"
                    if state[1] == -2:
                        if (state[0] == 1 and next_state[0] == -3) or (state[0] == -3 and next_state[0] == -1) or (state[0] == -1 and next_state[0] == 3) or (state[0] == 3 and next_state[0] == 1):
                            step = "R2"
                            step_ = "R4"
                        else:
                            step = "R4"
                            step_ = "R2"

                    if state[1] == 3:
                        if (state[0] == 1 and next_state[0] == -2) or (state[0] == -2 and next_state[0] == -1) or (state[0] == -1 and next_state[0] == 2) or (state[0] == 2 and next_state[0] == 1):
                            step = "R2"
                            step_ = "R4"
                        else:
                            step = "R4"
                            step_ = "R2"
                    if state[1] == -3:
                        if (state[0] == 1 and next_state[0] == 2) or (state[0] == 2 and next_state[0] == -1) or (state[0] == -1 and next_state[0] == -2) or (state[0] == -2 and next_state[0] == 1):
                            step = "R2"
                            step_ = "R4"
                        else:
                            step = "R4"
                            step_ = "R2"
                tran = ["L0", step, "L1", "R0", step_, "R1"]
                rot = l_rot(turn)

            # 左右 状态均不相同，两步调整
            else:
                if next_state[0] == result:
                    rot1 = l_rot(turn)
                    rot2 = r_rot(next_turn)
                else:
                    rot1 = r_rot(turn)
                    rot2 = l_rot(next_turn)

                # 左相反，同时右相反
                if state[0] + next_state[0] == 0 and state[1] + next_state[1] == 0:
                    step1 = "L3"
                    step1_ = "L5"
                    step2 = "R3"
                    step2_ = "R5"
                else:
                    # 左相反，右不相反
                    if state[0] + next_state[0] == 0:
                        step2 = "R3"
                        step2_ = "R5"
                        if state[0] == 1:
                            if (state[1] == 2 and next_state[1] == -3) or (state[1] == -3 and next_state[1] == -2) or (
                                    state[1] == -2 and next_state[1] == 3) or (state[1] == 3 and next_state[1] == 2):
                                step1 = "L2"
                                step1_ = "L4"
                            else:
                                step1 = "L4"
                                step1_ = "L2"
                        if state[0] == -1:
                            if (state[1] == 2 and next_state[1] == 3) or (state[1] == 3 and next_state[1] == -2) or (
                                    state[1] == -2 and next_state[1] == -3) or (state[1] == -3 and next_state[1] == 2):
                                step1 = "L2"
                                step1_ = "L4"
                            else:
                                step1 = "L4"
                                step1_ = "L2"

                        if state[0] == 2:
                            if (state[1] == 1 and next_state[1] == 3) or (state[1] == 3 and next_state[1] == -1) or (
                                    state[1] == -1 and next_state[1] == -3) or (state[1] == -3 and next_state[1] == 1):
                                step1 = "L2"
                                step1_ = "L4"
                            else:
                                step1 = "L4"
                                step1_ = "L2"
                        if state[0] == -2:
                            if (state[1] == 1 and next_state[1] == -3) or (state[1] == -3 and next_state[1] == -1) or (
                                    state[1] == -1 and next_state[1] == 3) or (state[1] == 3 and next_state[1] == 1):
                                step1 = "L2"
                                step1_ = "L4"
                            else:
                                step1 = "L4"
                                step1_ = "L2"

                        if state[0] == 3:
                            if (state[1] == 1 and next_state[1] == -2) or (state[1] == -2 and next_state[1] == -1) or (
                                    state[1] == -1 and next_state[1] == 2) or (state[1] == 2 and next_state[1] == 1):
                                step1 = "L2"
                                step1_ = "L4"
                            else:
                                step1 = "L4"
                                step1_ = "L2"
                        if state[0] == -3:
                            if (state[1] == 1 and next_state[1] == 2) or (state[1] == 2 and next_state[1] == -1) or (
                                    state[1] == -1 and next_state[1] == -2) or (state[1] == -2 and next_state[1] == 1):
                                step1 = "L2"
                                step1_ = "L4"
                            else:
                                step1 = "L4"
                                step1_ = "L2"
                    # 右相反，左不相反
                    elif state[1] + next_state[1] == 0:
                        step1 = "L3"
                        step1_ = "L5"
                        # 由于先调右，所以调左时状态相反
                        if state[1] == -1:
                            if (state[0] == 2 and next_state[0] == -3) or (state[0] == -3 and next_state[0] == -2) or (
                                    state[0] == -2 and next_state[0] == 3) or (state[0] == 3 and next_state[0] == 2):
                                step2 = "R2"
                                step2_ = "R4"
                            else:
                                step2 = "R4"
                                step2_ = "R2"
                        if state[1] == 1:
                            if (state[0] == 2 and next_state[0] == 3) or (state[0] == 3 and next_state[0] == -2) or (
                                    state[0] == -2 and next_state[0] == -3) or (state[0] == -3 and next_state[0] == 2):
                                step2 = "R2"
                                step2_ = "R4"
                            else:
                                step2 = "R4"
                                step2_ = "R2"

                        if state[1] == -2:
                            if (state[0] == 1 and next_state[0] == 3) or (state[0] == 3 and next_state[0] == -1) or (
                                    state[0] == -1 and next_state[0] == -3) or (state[0] == -3 and next_state[0] == 1):
                                step2 = "R2"
                                step2_ = "R4"
                            else:
                                step2 = "R4"
                                step2_ = "R2"
                        if state[1] == 2:
                            if (state[0] == 1 and next_state[0] == -3) or (state[0] == -3 and next_state[0] == -1) or (
                                    state[0] == -1 and next_state[0] == 3) or (state[0] == 3 and next_state[0] == 1):
                                step2 = "R2"
                                step2_ = "R4"
                            else:
                                step2 = "R4"
                                step2_ = "R2"

                        if state[1] == -3:
                            if (state[0] == 1 and next_state[0] == -2) or (state[0] == -2 and next_state[0] == -1) or (
                                    state[0] == -1 and next_state[0] == 2) or (state[0] == 2 and next_state[0] == 1):
                                step2 = "R2"
                                step2_ = "R4"
                            else:
                                step2 = "R4"
                                step2_ = "R2"
                        if state[1] == 3:
                            if (state[0] == 1 and next_state[0] == 2) or (state[0] == 2 and next_state[0] == -1) or (
                                    state[0] == -1 and next_state[0] == -2) or (state[0] == -2 and next_state[0] == 1):
                                step2 = "R2"
                                step2_ = "R4"
                            else:
                                step2 = "R4"
                                step2_ = "R2"
                    # 左右均不相反，先根据左调右再根据调完的右调左，调完后根据复原公式再依次复原
                    else:
                        if state[0] == 1:
                            if (state[1] == 2 and next_state[1] == -3) or (state[1] == -3 and next_state[1] == -2) or (
                                    state[1] == -2 and next_state[1] == 3) or (state[1] == 3 and next_state[1] == 2):
                                step1 = "L2"
                                step1_ = "L4"
                            else:
                                step1 = "L4"
                                step1_ = "L2"
                        if state[0] == -1:
                            if (state[1] == 2 and next_state[1] == 3) or (state[1] == 3 and next_state[1] == -2) or (
                                    state[1] == -2 and next_state[1] == -3) or (state[1] == -3 and next_state[1] == 2):
                                step1 = "L2"
                                step1_ = "L4"
                            else:
                                step = "L4"
                                step_ = "L2"

                        if state[0] == 2:
                            if (state[1] == 1 and next_state[1] == 3) or (state[1] == 3 and next_state[1] == -1) or (
                                    state[1] == -1 and next_state[1] == -3) or (state[1] == -3 and next_state[1] == 1):
                                step1 = "L2"
                                step1_ = "L4"
                            else:
                                step1 = "L4"
                                step1_ = "L2"
                        if state[0] == -2:
                            if (state[1] == 1 and next_state[1] == -3) or (state[1] == -3 and next_state[1] == -1) or (
                                    state[1] == -1 and next_state[1] == 3) or (state[1] == 3 and next_state[1] == 1):
                                step1 = "L2"
                                step1_ = "L4"
                            else:
                                step1 = "L4"
                                step1_ = "L2"

                        if state[0] == 3:
                            if (state[1] == 1 and next_state[1] == -2) or (state[1] == -2 and next_state[1] == -1) or (
                                    state[1] == -1 and next_state[1] == 2) or (state[1] == 2 and next_state[1] == 1):
                                step1 = "L2"
                                step1_ = "L4"
                            else:
                                step1 = "L4"
                                step1_ = "L2"
                        if state[0] == -3:
                            if (state[1] == 1 and next_state[1] == 2) or (state[1] == 2 and next_state[1] == -1) or (
                                    state[1] == -1 and next_state[1] == -2) or (state[1] == -2 and next_state[1] == 1):
                                step1 = "L2"
                                step1_ = "L4"
                            else:
                                step1 = "L4"
                                step1_ = "L2"

                        # 根据下个状态的右调左
                        if next_state[1] == 1:
                            if (state[0] == 2 and next_state[0] == -3) or (state[0] == -3 and next_state[0] == -2) or (
                                    state[0] == -2 and next_state[0] == 3) or (state[0] == 3 and next_state[0] == 2):
                                step = "R2"
                                step_ = "R4"
                            else:
                                step = "R4"
                                step_ = "R2"
                        if next_state[1] == -1:
                            if (state[0] == 2 and next_state[0] == 3) or (state[0] == 3 and next_state[0] == -2) or (
                                    state[0] == -2 and next_state[0] == -3) or (state[0] == -3 and next_state[0] == 2):
                                step = "R2"
                                step_ = "R4"
                            else:
                                step = "R4"
                                step_ = "R2"

                        if next_state[1] == 2:
                            if (state[0] == 1 and next_state[0] == 3) or (state[0] == 3 and next_state[0] == -1) or (
                                    state[0] == -1 and next_state[0] == -3) or (state[0] == -3 and next_state[0] == 1):
                                step = "R2"
                                step_ = "R4"
                            else:
                                step = "R4"
                                step_ = "R2"
                        if next_state[1] == -2:
                            if (state[0] == 1 and next_state[0] == -3) or (state[0] == -3 and next_state[0] == -1) or (
                                    state[0] == -1 and next_state[0] == 3) or (state[0] == 3 and next_state[0] == 1):
                                step = "R2"
                                step_ = "R4"
                            else:
                                step = "R4"
                                step_ = "R2"

                        if next_state[1] == 3:
                            if (state[0] == 1 and next_state[0] == -2) or (state[0] == -2 and next_state[0] == -1) or (
                                    state[0] == -1 and next_state[0] == 2) or (state[0] == 2 and next_state[0] == 1):
                                step = "R2"
                                step_ = "R4"
                            else:
                                step = "R4"
                                step_ = "R2"
                        if next_state[1] == -3:
                            if (state[0] == 1 and next_state[0] == 2) or (state[0] == 2 and next_state[0] == -1) or (
                                    state[0] == -1 and next_state[0] == -2) or (state[0] == -2 and next_state[0] == 1):
                                step = "R2"
                                step_ = "R4"
                            else:
                                step = "R4"
                                step_ = "R2"

                # 先调右再调左
                tran = ["R0", step1, "R1", "L0", step1_, step2, "L1", "R0", step2_, "R1"]
                rot = rot1 + rot2
        hole_tran = hole_tran + (tran + rot)
    return  hole_tran


def processResults2(result):
    results, turns = coverlist(result)
    results = tran_to_Num(results)
    # print(result)
    # print(results)
    # print(len(results))
    graph, path, point, end = Graph(results)
    start = 0

    G = {'V':point,
         'E':path}

    # distance, path_target = networkx_dijkstra(point, path, start, end)
    # pic_graph(point, path) #可视化图
    min_distance = len(path)*10
    min_path = []
    for p in end:
        distance, path_target = networkx_dijkstra(point, path, start, p)
        # path_target, distance = Dijkstra(start, p, G)
        # distance, path_target = Dijkstra_EX(point, path, start, p)
        # print(distance)
        # print(path_target)
        if distance < min_distance:
            min_distance = distance
            min_path = path_target
    # print(min_distance)
    # print(min_path)
    path_node = []
    for i in range(len(graph)):
        nodes = graph[i]
        for node in nodes:
            if node.No in min_path:
                #print(node.No)
                #print(node.state)
                path_node.append(node.state)

    # print(path_node)
    hole_step = step_tran(path_node, results, turns)
    # print(hole_step)
    return hole_step

'''
0 : 松开
1 : 夹紧
2 : 顺时针转90
3 : 顺时针转180
4 : 逆时针转90
5 : 逆时针转180
'''

'''
F\B = 1\-1;
U\D = 2\-2;
R\L = 3\-3
'''

def one_step_tran(tran):
    z = ["L0", "R2", "L1", "R0", "R4", "R1"]
    return z

def two_step_tran():
    z = ["L0", "R4", "L1", "R0", "R2", "L2", "L0", "L4", "L1"]
    return z

def sorts(tran):
    if tran == 'R0':
        tran = 'a'
        return tran
    if tran == 'R1':
        tran = 'b'
        return tran
    if tran == 'R2':
        tran = 'c'
        return tran
    if tran == 'R3':
        tran = 'd'
        return tran
    if tran == 'R4':
        tran = 'e'
        return tran
    if tran == 'R5':
        tran = 'f'
        return tran
    if tran == 'L0':
        tran = 'A'
        return tran
    if tran == 'L1':
        tran = 'B'
        return tran
    if tran == 'L2' :
        tran = 'C'
        return tran
    if tran == 'L3':
        tran = 'D'
        return tran
    if tran == 'L4':
        tran = 'E'
        return tran
    if tran == 'L5':
        tran = 'F'
        return tran



if __name__ == '__main__':
    # cube_position = "UDUDUDUDURLRLRLRLRFBFBFBFBFDUDUDUDUDLRLRLRLRLBFBFBFBFB"
    cube_position = "DRLUUBFBRBLURRLRUBLRDDFDLFUFUFFDBRDUBRUFLLFDDBFLUBLRBD"
    result = kociemba.solve(cube_position)
    # result = "R U F L R U R B U F U F D F D F D U B"
    hole_step = processResults2(result)
    print(len(hole_step))
    print(hole_step)
    #print(G[1][0].path)
