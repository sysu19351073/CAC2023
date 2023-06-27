import kociemba

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

def processResults(cube_position):
    # 将 kociemba 解处理为需要的形式
    # 将魔方复原公式转为机械夹爪的工作公式
    trans = cube_position.split()
    tran = []
    formula = []
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

        if tran1 == "U":
            rot = l_rot(tran2)
            if pretran == "D":
                tran = b()
            if pretran == "B":
                tran = a()
            if pretran == "R":
                tran = p()
            if pretran == "F":
                tran = w()
            if pretran == "L":
                tran = y()

        if tran1 == "R":
            rot = r_rot(tran2)
            if pretran == "F":
                tran = g()
            if pretran == "L":
                tran = h()
            if pretran == "D":
                tran = i()
            if pretran == "B":
                tran = o()
            if pretran == "U":
                tran = q()

        if tran1 == "F":
            rot = r_rot(tran2)
            if pretran == "R":
                tran = ee()
            if pretran == "L":
                tran = g()
            if pretran == "D":
                tran = j()
            if pretran == "B":
                tran = r()
            if pretran == "U":
                tran = t()

        if tran1 == "D":
            rot = l_rot(tran2)
            if pretran == "B":
                tran = c()
            if pretran == "U":
                tran = d()
            if pretran == "R":
                tran = i()
            if pretran == "F":
                tran = k()
            if pretran == "L":
                tran = m()

        if tran1 == "L":
            rot = r_rot(tran2)
            if pretran == "R":
                tran = f()
            if pretran == "F":
                tran = ee()
            if pretran == "D":
                tran = l()
            if pretran == "B":
                tran = s()
            if pretran == "U":
                tran = u()

        if tran1 == "B":
            rot = l_rot(tran2)
            if pretran == "D":
                tran = a()
            if pretran == "U":
                tran = c()
            if pretran == "R":
                tran = n()
            if pretran == "F":
                tran = v()
            if pretran == "L":
                tran = x()

        for z in rot:
            tran.append(z)

        for z in tran:
            formula.append(z)

    # cube_position = " ".join(cube_position)  # 连为一个string
    return formula

def r_rot(tran):
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

def a():
    z = ["L0", "R2", "L1", "R0", "R4", "R1"]
    return z

def b():
    z = ["L0", "R3", "L1", "R0", "R5", "R1"]
    return z

def c():
    z = ["L0", "R4", "L1", "R0", "R2", "R1"]
    return z

def d():
    z = ["L0", "R5", "L1", "R0", "R3", "R1"]
    return z

def ee():
    z = ["R0", "L2", "R1", "L0", "L4", "L1"]
    return z

def f():
    z = ["R0", "L3", "R1", "L0", "L5", "L1"]
    return z

def g():
    z = ["R0", "L4", "R1", "L0", "L2", "L1"]
    return z

def h():
    z = ["R0", "L5", "R1", "L0", "L3", "L1"]
    return z

def i():
    z = []
    return z

def j():
    return ee()

def k():
    return j()

def l():
    return f()

def m():
    return h()

def n():
    return a()

def o():
    return c()

def p():
    return b()

def q():
    return d()

def r():
    z = c()
    for i in ee():
        z.append(i)
    z = ["L0", "R4", "L1", "R0", "R2", "L2", "R1", "L0", "L4", "L1"]
    return  z

def s():
    z = c()
    for i in f():
        z.append(i)
    z = ["L0", "R4", "L1", "R0", "R2", "L3", "R1", "L0", "L5", "L1"]
    return z

def t():
    z = d()
    for i in ee():
        z.append(i)
    z = ["L0", "R5", "L1", "R0", "R3", "L2", "R1", "L0", "L4", "L1"]
    return z

def u():
    z = d()
    for i in f():
        z.append(i)
    z = ["L0", "R5", "L1", "R0", "R3", "L3", "R1", "L0", "L5", "L1"]
    return z

def v():
    z = g()
    for i in a():
        z.append(i)
    z = ["R0", "L4", "R1", "L0", "L2", "R2", "L1", "R0", "R4", "R1"]
    return z

def w():
    z = g()
    for i in b():
        z.append(i)
    z = ["R0", "L4", "R1", "L0", "L2", "R3", "L1", "R0", "R5", "R1"]
    return z

def x():
    z = h()
    for i in a():
        z.append(i)
    z = ["R0", "L5", "R1", "L0", "L3", "R2", "L1", "R0", "R4", "R1"]
    return z

def y():
    z = h()
    for i in b():
        z.append(i)
    z = ["R0", "L5", "R1", "L0", "L3", "R3", "L1", "R0", "R5", "R1"]
    return z

if __name__ == '__main__':
    # cube_position = "UDUDUDUDURLRLRLRLRFBFBFBFBFDUDUDUDUDLRLRLRLRLBFBFBFBFB"
    # cube_position = "DRLUUBFBRBLURRLRUBLRDDFDLFUFUFFDBRDUBRUFLLFDDBFLUBLRBD"
    # result = kociemba.solve(cube_position)
    result = "R U F L R U R B U F U F D F D F D U B"
    lens = len(result.split())
    results = len(processResults(result))
    print(result)
    print(lens)
    print(results)