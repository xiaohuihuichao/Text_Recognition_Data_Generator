import cv2
import random
import numpy as np
from itertools import product

# 随机画线
# 最多n条线
def random_line_1(img, n=2):
    img_h, img_w = img.shape[0:2]

    n = np.random.randint(1, n)
    for _ in range(n):
        # 随机位置， 随机颜色， 随机宽度
        line_color = random_text_color()
        line_thickness = np.random.randint(1, 2)
        if np.random.uniform(0, 1) < 0.5:
            # 竖线
            i = np.random.randint(0, img_w)
            point_1 = (i, 0)
            point_2 = (i, img_h)
        else:
            # 横线
            i = np.random.randint(0, img_h)
            point_1 = (0, i)
            point_2 = (img_w, i)
        img = cv2.line(img, point_1, point_2, line_color, line_thickness)
    return img
    
    
def random_line_2(img, text_RGB, n=2):
    text_height, text_width = img.shape[0:2]
    n = np.random.randint(0, n)
    for _ in range(n):
        pts = [(np.random.randint(0, text_width), 0),
                (text_width, np.random.randint(0, text_height)),
                (np.random.randint(0, text_width), text_height),
                (0, np.random.randint(0, text_height))]
        p1, p2 = random.sample(pts, 2)
        img = cv2.line(img, p1, p2, (text_RGB))
    return img
    
    
# 随机画椭圆
def draw_ellipse(img, text_size, n=2):
    n = np.random.randint(1, n)
    for _ in range(n):
        axes_short = np.random.randint(15, text_size)
        axes_long = np.random.randint(text_size*3, text_size*4)
        axes = (axes_short, axes_long)

        center = (np.random.randint(0, text_size*3), np.random.randint(0, text_size))
        center = (text_size*3, text_size)
        angle = np.random.randint(75, 125)
        color = random.choice([(np.random.randint(0, 70), np.random.randint(0, 70), np.random.randint(200, 255)), # 接近红色
                                random_RBG(0, 70)]) # 接近黑色
        thickness = np.random.randint(1, 2)
        img = cv2.ellipse(img, center, axes, angle, 0, 360, color, thickness=thickness)
    return img


# 椒盐噪声
def noise(img, snr=0.95):
    h, w = img.shape[0:2]
    num_noise = int(h*w*(1-snr))
    for _ in range(num_noise):
        rand_row = np.random.randint(1, h-1)
        rand_col = np.random.randint(1, w-1)

        if np.random.uniform(0, 1) < 0.5:
            color = np.asarray(random_RBG(0, 100), dtype=np.uint8)
        else:
            color = np.asarray(random_RBG(200, 255), dtype=np.uint8)

        img[rand_row, rand_col] = color
    return img


def perspective(img, points_from, points_to, out_wh, borderValue):
    M = cv2.getPerspectiveTransform(points_from, points_to)
    return cv2.warpPerspective(img, M, out_wh, borderValue=borderValue, flags=cv2.INTER_LINEAR)

def get_h(h, min_h_rate):
    h_min, h_max = 0, 0
    while h_max-h_min < h*min_h_rate:
        h_min, h_max = np.sort([np.random.randint(0, h), np.random.randint(0, h)])
    return h_min, h_max

def line_skew(img, borderValue, min_h_rate=0.5, w_offset=10):
    h, w = img.shape[0:2]
    start_h1, start_h2 = get_h(h, min_h_rate)
    end_h1, end_h2 = get_h(h, min_h_rate)
    points_to = np.asarray([[np.random.randint(0, w_offset), start_h1],
                            [np.random.randint(0, w_offset), start_h2],
                            [np.random.randint(w-w_offset, w), end_h1],
                            [np.random.randint(w-w_offset, w), end_h2]], dtype=np.float32)
    row_min = np.random.choice([round(min([start_h1, start_h2, end_h1, end_h2])), 0])
    row_max = np.random.choice([round(max([start_h1, start_h2, end_h1, end_h2])), -1])
    
    points_from = np.asarray([[0, 0], [0, h-1], [w-1, 0], [w-1, h-1]], dtype=np.float32)
    return perspective(img, points_from, points_to, (w, h), borderValue)[row_min:row_max]


def distortion_w(img, back_value, num_split_max=4, offset_max=0.1):
    h, w = img.shape[0:2]
    num_split = np.random.randint(2, num_split_max)
    middle_w = [np.random.uniform(-offset_max, offset_max)*w+w/num_split for i in range(num_split)]
    middle_w = [round(i) for i in middle_w]
    img_list = []
    for i in range(num_split):
        p_from = np.asarray([[w/num_split*i, 0],
                             [w/num_split*i, h],
                             [round(w/num_split*(i+1)), 0],
                             [round(w/num_split*(i+1)), h]], dtype=np.float32)
        p_to = np.asarray([[0, 0],
                           [0, h],
                           [middle_w[i], 0],
                           [middle_w[i], h]], dtype=np.float32)
        img_list += [perspective(img, p_from, p_to, (middle_w[i], h), back_value)]
    img = np.concatenate(img_list, axis=1)
    img = cv2.resize(img, dsize=(w, h))
    return img


def sin_distortion_h(img, back_value, num_peak_max=2, p_max=8):
    h, w = img.shape[0:2]
    
    num_peak = np.random.uniform(1, num_peak_max)
    p = np.random.uniform(1, p_max)
    result = np.zeros((round(h+p*2), w, 3), dtype=np.uint8) + np.asarray(back_value, dtype=np.uint8)
    def sin_h(x, start_theta=np.random.uniform(0, 2*np.pi)):
        return np.sin(x/w*np.pi*num_peak+start_theta) * p + p
        
    for y in range(h):
        for x in range(w):
            result[round(y+sin_h(x)), x] = img[y, x]
    return result


def distance(p1, p2):
    return np.sqrt(((p1-p2)**2).sum(axis=2))
    
def bright_point(img, r_max=0.7):
    h, w = img.shape[0:2]
    
    r = np.random.uniform(0, r_max)
    grid = np.asarray([[i, j] for i, j in product(range(h), range(w))]).reshape(h, w, 2)
    light_center = np.random.rand(1, 1, 2) * np.asarray([[h, w]])
    d = distance(grid, light_center) ** r
    m = d.max()
    d /= m
    weight = 1 - d
    weight = np.stack([weight, weight, weight], axis=2)
    
    bright = np.zeros_like(img) + 255
    bright_point = np.asarray(weight*bright + (1-weight)*img, dtype=np.uint8)
    return bright_point


def random_RBG(s, e):
    return (np.random.randint(s, e),
            np.random.randint(s, e),
            np.random.randint(s, e))
    
def random_color(i1, i2, s):
    R = np.random.randint(i1, i2)
    G = np.random.randint(max(0, R-s), min(R+s, 255))
    B = np.random.randint(max(0, R-s), min(R+s, 255))
    return (R, G, B)
def random_text_color(i1=0, i2=70, s=15):
    return random_color(i1, i2, s)
def random_bg_color(i1=200, i2=255, s=15):
    return random_color(i1, i2, s)
