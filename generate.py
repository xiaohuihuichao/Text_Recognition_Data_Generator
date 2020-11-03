import os
import glob
import random
from itertools import product

import cv2
import numpy as np
from PIL import Image, ImageFont, ImageDraw

from tools import random_line_1, random_line_2
from tools import draw_ellipse, noise, line_skew
from tools import distortion_w, sin_distortion_h
from tools import bright_point, random_text_color, random_bg_color


class write:
    def __init__(self, font_dir, text_size):
        font_files = glob.glob(os.path.join(f"{font_dir}", "*"))
        self.fonts = [ImageFont.truetype(font_file, text_size) for font_file in font_files]
        self.text_size = text_size

    def write(self, text, text_RGB=(0, 0, 0), bg_RGB=(255, 255, 255)):
        # 图像增强
        line_rate_1 = 0.02
        line_rate_2 = 0.02
        ellipse_rate = 0.2
        gaussian_rate = 0.3
        noise_rate = 0.8
        bright_point_rate = 0.3
        skew_rate = 0.2
        sin_distortion_rate = 0.1
        distortion_w_rate= 0.1

        font = np.random.choice((self.fonts))
        text_width, text_height = font.getsize(text)

        h_extend = np.random.randint(0, self.text_size)
        text_width += h_extend
        h_start = np.random.randint(0, h_extend) if h_extend > 0 else 0

        v_extend = np.random.randint(0, self.text_size//4)
        text_height += v_extend
        v_start = np.random.randint(0, v_extend) if v_extend > 0 else 0
        
        canvas = Image.new("RGB", [text_width, text_height], bg_RGB)
        draw = ImageDraw.Draw(canvas)
        draw.text((h_start, v_start), text, font=font, fill=text_RGB, spaceing=np.random.randint(0, 4))
        
        img = cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)
        img = cv2.GaussianBlur(img, (3, 3), np.random.randint(0, 5))        
        
        
        # 随机划横竖线
        if np.random.uniform(0, 1) < line_rate_1:
            img = random_line_1(img)
            
        # 随机划线
        if np.random.uniform(0, 1) < line_rate_2:
            img = random_line_2(img, text_RGB)
        
        # 随机画椭圆
        if np.random.uniform(0, 1) < ellipse_rate:
            img = draw_ellipse(img, self.text_size)
            
        # 灯光白点反光效果
        if np.random.uniform(0, 1) < bright_point_rate:
            img = bright_point(img)
            
        # 透视变换做倾斜
        if np.random.uniform(0, 1) < skew_rate:
            img = line_skew(img, bg_RGB)
            
        # sin弯曲
        if np.random.uniform(0, 1) < sin_distortion_rate:
            img = sin_distortion_h(img, bg_RGB)
        
        # 水平拉伸
        if np.random.uniform(0, 1) < distortion_w_rate:
            img = distortion_w(img, bg_RGB)
        
        # 椒盐噪声
        if np.random.uniform(0, 1) < noise_rate:
            img = noise(img, np.random.uniform(0.95, 1))
        
        # 高斯模糊
        if np.random.uniform(0, 1) < gaussian_rate:
            kernel_size = [(3, 3), (5, 5),  # 模糊
                           (1, 3), #(1, 5),  # 上下运动模糊
                           (3, 1), #(5, 1),  # 左右运动模糊
                           ]
            kernel_size = random.choice(kernel_size)
            img = cv2.GaussianBlur(img, kernel_size, np.random.randint(0, 10))
        
        h, w = img.shape[0:2]
        w = 32 * w / h
        img = cv2.resize(img, dsize=(round(w), 32))
        return img


def imshow(win_name, img, t=0):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, img)
    return cv2.waitKey(t)


if __name__ == "__main__":
    text = "中华人民共和国机动车行驶证副页"
    
    font_dir = "./font_files"
    text_size = 32
    
    img_path = "test.jpg"
    
    w = write(font_dir, text_size)
    while True:
        bg_color = random_bg_color()
        text_color = random_text_color()
        img = w.write(text, text_color, bg_color)
        cv2.imwrite(img_path, img)
        imshow("img", img, 1)
    