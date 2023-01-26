import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
from pickHSV import HSVpicker# 导入阈值选择器
from Kalman import KFT# 导入卡尔曼滤波模型
from SamBox import SampleBox

iswrite = False#结果视频是否保存写入
txtw = False# 估计轨迹是否输出到txt
BOX = True# 是否开启采样框

savename = "ResultVideos/new_1.mp4"# 写入的路径
video = 'RawVideos/1.mp4'#需要分析的视频文件路径
#RawVideos/0.mp4 篮球
#RawVideos/0.mp4  字典遮挡
#以上是有标注的

#src/snk1.mp4  台球
#src/12test.mp4 两物体

if __name__ == '__main__':
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置图形里面中文为黑体
    n = int(input('请输入目标个数：'))
            #橘子                                       瓶盖1                                  cxk                             瓶盖2
    #obj = [[[0, 142, 72], [21, 255, 255]], [[80, 84, 163], [123, 146, 249]], [[ 0,  67, 108], [85, 255, 236]],[[76,  56, 191], [132, 123, 255]]]# 这是事先调好的,橘子和瓶盖
    obj = []
    #保存首帧图像
    camera = cv2.VideoCapture(video)
    (ret, frame) = camera.read()
    cv2.imwrite('src/imgtemp.jpg', frame)
    camera.release()
    cv2.destroyAllWindows()
    #i = cv2.imread('imgtemp.jpg')
    n1 = 0
    print('请选择要跟踪的目标物体的阈值')
    while n1 < n:
        p = HSVpicker(source='src/imgtemp.jpg')
        p.begin()
        thres = p.getfinalhsv()
        obj.append(thres)
        n1 += 1
        print('为：', thres)
    # 设定阈值，HSV空间
    obj = [[[76,  56, 191], [132, 123, 255]], ]# 调好的阈值，可以注释掉
    num_of_objs = len(obj)#要追踪的物体个数

    OB = [[[], []], ]# 记录观测数据
    ES = [[[], []], ]# 记录估计数据

    s = [[], []]
    for i in range(num_of_objs - 1):
        OB.append(list(s))
        ES.append(list(s))

    redLower = np.array(obj[0][0])
    redUpper = np.array(obj[0][1])
    # 初始化追踪点的列表
    mybuffer = 64
    pts = deque(maxlen=mybuffer)
    # 打开摄像头
    camera = cv2.VideoCapture(video)  # 'test.mp4'
    # 保存视频
    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取视频的宽度
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 获取视频的高度
    print('视频 w h =', width, height)
    fps = camera.get(cv2.CAP_PROP_FPS)  # 获取视频的帧率
    fourcc = int(camera.get(cv2.CAP_PROP_FOURCC))  # 视频的编码
    # 定义视频对象输出
    if iswrite:
        writer = cv2.VideoWriter(savename, fourcc, fps, (width, height))#200
    # 等待两秒
    time.sleep(2)
    # 遍历每一帧，检测瓶盖
    kfts = []#各自的kft
    lastx = [0 for i in range(num_of_objs)]
    lasty = [0 for i in range(num_of_objs)]
    FPS = []# 用来计算平均帧率
    fn = 0# 帧号
    UNOB = [[] for i in range(num_of_objs)]# 记录未观测到数据的帧号
    boxs = []# 采样框们
    res = [[] for i in range(num_of_objs)]# 采样框的返回值们
    find = [1 for i in range(num_of_objs)]# 标记是否找到


    while True:
        # 读取帧
        fn += 1
        (ret, frame) = camera.read()
        if ret == False:
            fn -= 1
            break
        timer = cv2.getTickCount()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        for i in range(num_of_objs):
            if fn == 1:  # 如果是首帧
                find[i] = 1# 我们默认首帧中有目标
                [x, y, w, h] = cv2.selectROI('roi', frame, False, False)  # 预选一下采样框大小以及初始位置，如果这样设置，最好首帧就有完整目标，当然可以提前设置，不过只是个初始位置影响不大。
                print('roi位置x y w h =', [x, y, w, h])
                cv2.waitKey(0)
                xx = int(x + w//2)
                yy = int(y + h//2)
                x0 = np.array([xx, 0, yy, 0])  # 初始位置,根据框确定初始位置

                kft = KFT(X=x0)  # 申请一个KFT
                kfts.append(kft)  # 加入到追踪器列表 kfts
                lastx[i] = xx
                lasty[i] = yy

                OB[i][0].append(xx)
                OB[i][1].append(yy)
                ES[i][0].append(xx)
                ES[i][1].append(yy)
                # 采样框
                if BOX == True:
                    box = SampleBox(cx=xx, cy=yy, h=2 * h, w=2 * w, maxx=width, maxy=height)  # 每个追踪对象申请一个采样框
                    boxs.append(box)
                    boxs[i].set(cx=xx, cy=yy)  # 当前位置给一个采样框
                    re = boxs[i].getxyhwcxcy()
                    res[i] = re
                    print('采样框位置x y h w cx cy =', res)
                continue
            (X1, P1) = kfts[i].predict()  # 先做预测，再在预测的地方做观测，这句写到这不影响后面，反而符合采样框的逻辑

            # 根据阈值进行二值化
            mask = cv2.inRange(hsv, np.array(obj[i][0]), np.array(obj[i][1]))
            # 腐蚀操作
            mask = cv2.erode(mask, None, iterations=2)
            # 想去除背景小噪声可以适当增大腐蚀的迭代次数，对于大噪声不适用，
            # 大噪声或许需要ROI采样框来进行一定程度上的忽视，采样框应该难以解决相同物体颜色相遇遮挡时，
            # 这时需要对目标增加别的特征，如形状。
            # 若物体完全相同。目标特征基本相同时，阈值方法或许就不适用了。
            # 膨胀操作，其实先腐蚀再膨胀的效果是开运算，去除噪点
            mask = cv2.dilate(mask, None, iterations=2)

            if BOX == True:# 采样框
                if find[i] == 1:# 上一帧框内有观测
                    boxs[i].set(cx=int(X1[0]), cy=int(X1[2]))  # 在卡尔曼预测位置给一个采样框，在框里观测
                else:# 采样框丢失处理
                    boxs[i].boxlost()

                res[i] = boxs[i].getxyhwcxcy()
                mask_roi = mask[int(res[i][1]):int(res[i][1] + res[i][2]), int(res[i][0]):int(res[i][0] + res[i][3])].copy()
                xmin = int(res[i][0])  # x
                xmax = int(res[i][0] + res[i][3])  # x+w
                ymin = int(res[i][1])  # y
                ymax = int(res[i][1] + res[i][2])  # y+h
                #print((xmin, ymin), (xmax, ymax))
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 2)# 输出采样框
            else:
                mask_roi = mask.copy()

            # 轮廓检测
            cnts = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]#在采样框里检测
            # 初始化瓶盖圆形轮廓质心
            center = None

            # 如果存在轮廓,即观测到
            if len(cnts) > 0:
                find[i] = 1
                # 找到面积最大的轮廓
                c = max(cnts, key=cv2.contourArea)
                # 确定面积最大的轮廓的外接圆
                ((x, y), radius) = cv2.minEnclosingCircle(c)
                # 计算轮廓的矩
                M = cv2.moments(c)
                # 计算质心
                center = [int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])]  # center为观测值
                #此时的值是在roi里的值，如果是采样框，需要加上框的位置
                if BOX == True:
                    center[0] = center[0] + res[i][0]
                    center[1] = center[1] + res[i][1]
                    x = x + res[i][0]
                    y = y + res[i][1]

                OB[i][0].append(int(center[0]))
                OB[i][1].append(int(center[1]))

                vx = center[0] - lastx[i]
                vy = center[1] - lasty[i]
                z = np.array([center[0], vx, center[1], vy])
                (X, P, K) = kfts[i].update(Z=z)

                # X就是最优估计

                lastx[i] = center[0]
                lasty[i] = center[1]

                esX = X[0]
                esY = X[2]

                ES[i][0].append(int(esX))
                ES[i][1].append(int(esY))
                # 只有当半径大于10时，才执行画图
                if radius > 10:
                    cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)# 黄色
                    cv2.circle(frame, (int(center[0]), int(center[1])), 2, (0, 0, 255), -1)# 红色
                    cv2.circle(frame, (int(esX), int(esY)), 2, (0, 0, 0), -1)  # 最优估计,黑色
                    # 把质心添加到pts中，并且是添加到列表左侧
                    pts.appendleft(center)
            else:
                #未检测到 ，把预测值当观测值

                find[i] = 0# 标记置为0，表示该帧未搜索到，那么下一帧将全视野检索
                # 记录未观测到的帧号
                UNOB[i].append(fn)

                vx = X1[1]
                vy = X1[3]
                z = np.array([lastx[i] + vx, vx, lasty[i] + vy, vy])
                (X, P, K) = kfts[i].update(Z=z)
                # X就是最优估计
                lastx[i] = lastx[i] + vx
                lasty[i] = lasty[i] + vy

                esX = X[0]
                esY = X[2]

                if esX < 0:# 预测不合理则用边界值代替
                    esX = 0
                if esY < 0:
                    esY = 0
                if esX > width:
                    esX = width
                if esY > height:
                    esY = height

                ES[i][0].append(int(esX))
                ES[i][1].append(int(esY))
                cv2.circle(frame, (int(esX), int(esY)), 2, (255, 0, 0), -1)  # 最优估计
                cv2.putText(frame, "Target Lost! ", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 1)
                pass
            """
                    for i in range(1, len(pts)):
                if pts[i - 1] is None or pts[i] is None:
                    continue
                # 计算所画小线段的粗细
                thickness = int(np.sqrt(mybuffer / float(i + 1)) * 2.5)
                # 画出小线段
                cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
            # res = cv2.bitwise_and(frame, frame, mask=mask)
            """
        # 遍历追踪点，画出轨迹
        fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)
        cv2.putText(frame, "FPS : " + str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (50, 170, 50), 2)
        # cv2.putText(frame, ":kalman est " + str(int(fps)), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)
        cv2.namedWindow('Frame', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.imshow('Frame', frame)
        FPS.append(fps)
        if iswrite == True:
            writer.write(frame)  # 视频保存
            print('结果视频写入成功')
        # 键盘检测，检测到esc键退出
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
    # 摄像头释放
    camera.release()
    # 销毁所有窗口
    cv2.destroyAllWindows()
    print('平均FPS:{}'.format(sum(FPS)//len(FPS)))

    #print('x轴最大观测值', max(OB[i][0]))
    #print('y轴最大观测值', max(OB[i][1]))
    if txtw == True:
        for j in range(len(ES)):
            f = open('ESpoints/new_E{}.txt'.format(j), 'w')
            for i in range(len(ES[j][0])):
                f.write(str(i + 1) + ' ' + str(ES[j][0][i]) + ' ' + str(ES[j][1][i]) + '\n')
            f.close()
        print('txt写入成功')
    print('帧数：', fn)
    #print(len(ES[0][0]))
    for i in range(num_of_objs):
        print('未观测到的帧号：', UNOB[i])
    #print('ES点数：', len(ES[0][0]))

    plt.figure('tracking')
    for i in range(num_of_objs):
        plt.ylim(height, 0)
        plt.plot(OB[i][0], OB[i][1], '.', label='观测轨迹{}'.format(i))
        plt.plot(ES[i][0], ES[i][1], '.', label='Kalman估计轨迹{}'.format(i))
        plt.plot([ES[i][0][x] for x in UNOB[i]], [ES[i][1][x] for x in UNOB[i]], '*', label='无观测时估计轨迹{}'.format(i))
        #plt.legend(['观测轨迹', 'Kalman估计轨迹'])
    plt.legend()
    plt.grid(ls='-.')
    plt.show()


