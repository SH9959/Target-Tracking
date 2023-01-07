# -*- coding:utf-8 -*-
import cv2
import numpy as np
"""
功能：读取一张图片，显示出来，转化为HSV色彩空间
     并通过滑块调节HSV阈值，实时显示
"""
ob = [[[0, 142, 72], [21, 255, 255]], [[80, 84, 163], [123, 146, 249]], [[76,  56, 191], [132, 123, 255]]]#一些探索好的阈值

class HSVpicker():# 定义一个HSVpicker类，初始化参数为一幅图像，经过调节后可输出满意的hsv阈值
    cnt = 0
    def __init__(self, source = 'f154.png'):
        self.id = HSVpicker.cnt
        HSVpicker.cnt += 1
        self.source = source
        self.source_resized = self.source
        self.hsv_low = np.array([0, 0, 0])
        self.hsv_high = np.array([0, 0, 0])
        self.finalhsv = np.array([self.hsv_low, self.hsv_high])

    def h_low(self, value):
        self.hsv_low[0] = value
    def h_high(self, value):
        self.hsv_high[0] = value
    def s_low(self, value):
        self.hsv_low[1] = value
    def s_high(self, value):
        self.hsv_high[1] = value
    def v_low(self, value):
        self.hsv_low[2] = value
    def v_high(self, value):
        self.hsv_high[2] = value

    def ChangeSize(self, SIZE = 600):#后来发现cv2.namedWindow('name', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)可以解决大小问题
        #SIZE = 600
        img_array = cv2.imread(self.source)
        new_image = cv2.resize(img_array, (SIZE, SIZE))
        self.source_resized = new_image
        #cv2.imwrite('s12.jpg', new_image)  # , [int(cv2.IMWRITE_JPEG_QUALITY),95])

    def begin(self):
        self.ChangeSize()
        image = self.source_resized # 读取一张resize后的图片，因为太大装不下
        #cv2.namedWindow('img', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow("img", image) # 显示图片

        cv2.namedWindow('image', 0)
        cv2.createTrackbar('H low', 'image', ob[2][0][0], 255, self.h_low)
        cv2.createTrackbar('H high', 'image', ob[2][1][0], 255, self.h_high)
        cv2.createTrackbar('S low', 'image', ob[2][0][1], 255, self.s_low)
        cv2.createTrackbar('S high', 'image', ob[2][1][1], 255, self.s_high)
        cv2.createTrackbar('V low', 'image', ob[2][0][2], 255, self.v_low)
        cv2.createTrackbar('V high', 'image', ob[2][1][2], 255, self.v_high)

        while True:
            dst = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # BGR转HSV
            dst = cv2.inRange(dst, self.hsv_low, self.hsv_high)  # 通过HSV的高低阈值，提取图像部分区域
            cv2.imshow('dst', dst)
            if (cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) & 0xFF == 27):
                print('{}号阈值选择完毕：'.format(HSVpicker.cnt))
                break

        self.finalhsv = np.array([self.hsv_low, self.hsv_high])
        cv2.destroyAllWindows()

    def getfinalhsv(self):
        return self.finalhsv

if __name__ == '__main__':
    p = HSVpicker(source = 'f154.png')
    p.begin()
    thres = p.getfinalhsv()
    print(thres)




