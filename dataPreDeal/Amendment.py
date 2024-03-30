import math
import cv2
from sympy import symbols, Eq, solve
# 240.25 365.4
import numpy as np

import numpy as np
from scipy.optimize import minimize


def get_p_to_l_distance(point, k, b):
    return (k * point[0] - point[1] + b) ** 2 / (1 + k ** 2)


# 定义目标函数和约束条件
def objective(x, points):
    return get_p_to_l_distance(points[0], x[0], x[2]) + \
           get_p_to_l_distance(points[2], x[0], x[2]) + \
           get_p_to_l_distance(points[1], x[1], x[3]) + \
           get_p_to_l_distance(points[3], x[1], x[3])


# k1*k2+1=0
def constraint1(x):
    return x[0] * x[1] + 1




def get_init(points):
    line1 = points[0] + points[2]
    line2 = points[1] + points[3]
    k1 = (line1[3] - line1[1]) / (line1[2] - line1[0] + 0.00000001)
    k2 = (line2[3] - line2[1]) / (line2[2] - line2[0] + 0.00000001)
    b1 = line1[1] - k1 * line1[0]
    b2 = line2[1] - k2 * line2[0]
    return [k1, k2, b1, b2]


def find_intersection(line1, line2):
    '''
    :param line1: [x1, y1, x2, y2]
    :param line2: [x1, y1, x2, y2]
    :return: [px, py]
    '''
    # line1 = points[0] + points[2]
    # line2 = points[1] + points[3]
    assert (line1[0] - line1[2]) != 0 or (line2[0] - line2[2]) != 0, "两条直线没有交点"
    if line1[0] == line1[2]:
        m2 = (line2[1] - line2[3]) / (line2[0] - line2[2])
        b2 = line2[1] - m2 * line2[0]
        return line1[0], m2 * line1[0] + b2
    elif line2[0] == line2[2]:
        m1 = (line1[1] - line1[3]) / (line1[0] - line1[2])
        b1 = line1[1] - m1 * line1[0]
        return line2[0], m1 * line2[0] + b1
    m1 = (line1[1] - line1[3]) / (line1[0] - line1[2])  # k = (y1 - y2) / (x1- x2)
    m2 = (line2[1] - line2[3]) / (line2[0] - line2[2])
    b1 = line1[1] - m1 * line1[0]  # b = y - k*x
    b2 = line2[1] - m2 * line2[0]
    x = (b2 - b1) / (m1 - m2)
    y = m1 * x + b1
    return x, y


from PIL import Image, ImageDraw


def draw_image(old_points, amd_point, imagePath='../data/images/train1.png'):
    image = Image.open(imagePath)
    draw = ImageDraw.Draw(image)
    # colors = ['red', 'white', 'black', 'orange']
    i = 0
    for point in old_points:
        draw.point(point, fill='red')
        i += 1
    draw.point(amd_point, fill='black')
    image.show()


def draw_arc(amd_point, imagePath='../data/images/train1.png'):
    image = Image.open(imagePath)
    draw = ImageDraw.Draw(image)
    # colors = ['red', 'white', 'black', 'orange']
    draw.arc(amd_point, 0, 360, fill='red')
    image.show()


def get_a_b(points):
    # 262.44
    a = math.sqrt(math.pow(points[0][0] - points[2][0], 2) + math.pow(points[0][1] - points[2][1], 2)) / 2
    b = math.sqrt(math.pow(points[1][0] - points[3][0], 2) + math.pow(points[1][1] - points[3][1], 2)) / 2
    return a, b


# 绘制拟合后的两条直线
def y_func(x, k, b):
    return k * x + b


def draw_line(old_points, res):
    imagePath = '../data/images/train1.png'
    image = Image.open(imagePath)
    draw = ImageDraw.Draw(image)
    amd_points = []
    for i, p in enumerate(old_points):
        if i % 2 == 0:
            amd_points.append([p[0], y_func(p[0] + 1, res[0], res[2])])
        else:
            amd_points.append([p[0], y_func(p[0] + 1, res[1], res[3])])
    points = [amd_points[0] + amd_points[2], amd_points[1] + amd_points[3]]

    draw.line(points[0], fill='red')
    draw.line(points[1], fill='blue')
    image.show()


def find_angle(tan_value):
    angle = math.atan(tan_value)
    return math.degrees(angle)


def fit_line(points, p):
    # 初始化变量
    x0 = np.array(get_init(points))
    constraint2 = {'type': 'eq',
                   'fun': lambda x: np.array(x[0] * p[0] + x[2] - p[1]),
                   'jac': lambda x: np.array([p[0], 0.0, 1.0, 0.0])}
    constraint3 = {'type': 'eq',
                   'fun': lambda x: np.array(x[1] * p[0] + x[3] - p[1]),
                   'jac': lambda x: np.array([0.0, p[0], 0.0, 1.0])}
    solution = minimize(objective, x0, args=(points, ), method='SLSQP', constraints=[{'fun': constraint1, 'type': 'eq'},
                                                                    constraint2,
                                                                    constraint3,
                                                                    ])
    return solution.x[0], solution.x[1], solution.x[2], solution.x[3]  # k1, k2, b1, b2


# 获取修正后的坐标点
def fit_points(k1, k2, b1, b2, a, b, cp):
    ha = symbols('ha^2')
    eqa = Eq((k1 ** 2) * ha + ha, a ** 2)
    s1 = solve(eqa, ha)[0]
    hb = symbols('hb^2')
    eqb = Eq((k2 ** 2) * hb + hb, b ** 2)
    s2 = solve(eqb, hb)[0]
    s1 = math.sqrt(s1)
    s2 = math.sqrt(s2)
    x_list = [cp[0] - s1, cp[0] - s2, cp[0] + s1, cp[0] + s2]
    points = []
    for i, x in enumerate(x_list):
        if i % 2 == 0:
            points.append([x, k1 * x + b1])
        else:
            points.append([x, k2 * x + b2])
    return points


def draw_arc(imagePath, cp, axis, angle):
    image = cv2.imread(imagePath)
    # a, b为半短轴长
    image = cv2.ellipse(image, (int(cp[0]), int(cp[1])), (int(axis[0]), int(axis[1])), round(angle, 0), 0, 360,
                        (255, 255, 255),
                        1)
    # image = cv2.circle(image, (int(cp[0]), int(cp[1])), 2, color=(255, 0, 0))
    cv2.imshow("image", image)
    # cv2.ellipse(img, (60, 20), (60, 20), 0, 0, 360, (255, 255, 255), 2);
    cv2.waitKey()
    cv2.destroyAllWindows()


def get_info(points):
    p = find_intersection(points[0] + points[2], points[1] + points[3])  # 获取椭圆交点
    a, b = get_a_b(points)
    k1, k2, b1, b2 = fit_line(points, p)
    if b > a:
        angle = find_angle(k2)
    else:
        angle = find_angle(k1)
    fitPoints = fit_points(k1, k2, b1, b2, a, b, p)
    return {'center': (round(p[0], 0), round(p[1], 0)), "angle": round(angle, 1), 'axis': (a, b),
            'axis_func': (k1, k2, b1, b2), 'fit_points': fitPoints}


if __name__ == '__main__':
    imagePath = '../data/images/train119.png'
    # 求a、b
    # points = [[375.1, 444.0], [375.3, 433.8], [392.7, 430.7], [387.7, 445.5]]
    points = [[393.6, 477.9], [412.0, 492.9]]
    # # info = get_info(points)
    # # print(info)
    draw_arc(imagePath, (381.0, 439.0), axis=(11.030072529226622, 8.524230170519786), angle=-40.5)
    draw_image(points, (381.0, 439.0), imagePath=imagePath)