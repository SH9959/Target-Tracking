import cv2

class SampleBox():#目的是，输入一个中心坐标以及宽和高，可以输出一个框（抽象的框）
    cnt = 0
    def __init__(self, x=10, y=10, h=200, w=150, cx=5, cy=5, maxx=1280, maxy=720):
        self.id = SampleBox.cnt
        SampleBox.cnt += 1
        self.x = x
        self.y = y
        self.h = h
        self.w = w
        self.cx = cx
        self.cy = cy
        self.mx = maxx
        self.my = maxy

        self.hh = h#先记录，避免hw在变换的过程中被修改
        self.ww = w
        print('{}号采样框选择完毕：'.format(SampleBox.cnt))
    def set(self, cx, cy):

        self.cx = cx
        self.cy = cy
        #self.h = h
        #self.w = w#用于采样框丢失

        self.x = self.cx - self.ww // 2
        self.y = self.cy - self.hh // 2
        #print(self.x, self.y)

        xflg = 0
        yflg = 0

        if self.x < 0:#左越界
            self.x = 0
            self.w = self.cx + (self.ww // 2) + 1
            xflg = 1
            #print(self.x, self.w)

        if self.cx + self.w // 2 + 1 > self.mx:#右越界
            self.w = self.mx - self.cx + self.ww // 2 - 1
            xflg = 1

        if self.y < 0:#上越界
            self.y = 0
            self.h = self.cy + (self.hh // 2) + 1
            yflg = 1

        if self.cy + self.h // 2 + 1 > self.my:#下越界
            self.h = self.my - self.cy + self.hh // 2 - 1
            yflg = 1

        if self.cx < 0:
            self.cx = 0
            xflg = 1

        if self.cx > self.mx:
            self.cx = self.mx
            xflg = 1

        if self.cy < 0:
            self.cy = 0
            yflg = 1

        if self.cy > self.my:
            self.cy = self.my
            yflg = 1

        if xflg == 0:
            self.w = self.ww# 恢复

        if yflg == 0:
            self.h = self.hh# 恢复

    def boxlost(self):#全视野搜索
        #self.set(cx=self.mx // 2, cy=self.my // 2)
        self.h = self.my
        self.w = self.mx
        self.x = 0
        self.y = 0
        self.cx = self.mx // 2
        self.cy = self.my // 2

        #print([self.cx, self.cy, self.h, self.w])
    def getxyhwcxcy(self):
        return [self.x, self.y, self.h, self.w, self.cx, self.cy]

if __name__ == "__main__":
    img = cv2.imread('src/imgtemp.jpg')

    h, w, d = img.shape
    print(h, w, d)
    box = SampleBox(cx=200, cy=20, h=200, w=150, maxx=w, maxy=h)

    for i in range(8):
        box.set(cx=i*180+10, cy=i*100+20)
        re = box.getxyhwcxcy()
        print(re)

        xmin = re[0]       #x
        xmax = re[0]+re[3] #x+w
        ymin = re[1]       #y
        ymax = re[1]+re[2] #y+h
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)

    cv2.imshow('src', img)
    cv2.waitKey()
