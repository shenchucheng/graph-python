#!/usr/bin/env python3
# -*- coding:UTF-8 -*-
# File Name: graph/sketch.py
# Author: Shechucheng
# Created Time: 2020-08-26 22:03:21


import cv2
import numpy as np
from numpy import pi, cos, sin, sqrt
from PIL import Image
    

def sketch(path, k=15,):
    img = cv2.imread(path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_blur = cv2.GaussianBlur(img_gray, ksize=(k, k),sigmaX=0, sigmaY=0)
    img_out=cv2.divide(img_gray, img_blur, scale=255)
    return Image.fromarray(img_out)


def sketch1(path, threshold=10):
    img = Image.open(path)
    if threshold < 0:
        threshold = 0
    if threshold > 100:
        threshold = 100
    width, height = img.size
    img = img.convert('L')  # 转为灰度图
    pixel = img.load()  # 获取灰度值
    for w in range(width):
        for h in range(height):
            if w == width - 1 or h == height - 1:
                continue
            xy = pixel[w, h]
            x1y1 = pixel[w + 1, h + 1]
            diff = abs(xy - x1y1)
            if diff >= threshold:
                pixel[w, h] = 0 #灰度越大越白，代表是轮廓
            else:
                pixel[w, h] = 255 #灰度越大越白，代表是轮廓
    return img


def sketch2(path, rate=0.1, ):
    img = np.asarray(Image.open(path).convert('L')).astype('float')
    grad_x, grad_y = np.gradient(img)
    grad_x, grad_y = grad_x*rate, grad_y*rate
    A = sqrt(grad_x**2 + grad_y**2 + 1)
    uni_x, uni_y, uni_z = grad_x/A, grad_y/A, 1/A
    vec_el, vec_az = pi/2.2, pi/4
    dx, dy, dz = cos(vec_el)*cos(vec_az), cos(vec_el)*sin(vec_az), sin(vec_el)
    b = 255*(dx*uni_x + dy*uni_y + dz*uni_z).clip(0, 255)
    return Image.fromarray(b.astype('uint8'))


def main():
    pass


if __name__ == "__main__":
    main()
     
