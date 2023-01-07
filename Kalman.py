import numpy as np
class KFT():
    """
    一个卡尔曼滤波追踪器KalmanFilterTracker
    """
    count = 0

    def __init__(self, A=np.array([[1, 1, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 1, 1],
                                   [0, 0, 0, 1]]),
                 B=np.array([0.5, 1, 0.5, 1]),
                 R=np.array([[0.1, 0, 0, 0],
                             [0, 0.1, 0, 0],
                             [0, 0, 0.1, 0],
                             [0, 0, 0, 0.1]]),
                 Q=np.array([[0.1, 0, 0, 0],
                             [0, 0.1, 0, 0],
                             [0, 0, 0.1, 0],
                             [0, 0, 0, 0.1]]),
                 H=np.eye(4),
                 U=np.array([0, 0, 0, 0]),
                 P=np.array([[0.1, 0, 0, 0],
                             [0, 0.1, 0, 0],
                             [0, 0, 0.1, 0],
                             [0, 0, 0, 0.1]]),
                 P1=np.array([[0.1, 0, 0, 0],
                              [0, 0.1, 0, 0],
                              [0, 0, 0.1, 0],
                              [0, 0, 0, 0.1]]),
                 K=np.array([0]),
                 X=np.array([0, 0, 0, 0]),
                 X1=np.array([0, 0, 0, 0])):
        self.id = KFT.count
        KFT.count += 1
        self.A = A
        self.B = B
        self.R = R
        self.Q = Q
        self.H = H
        self.U = U

        self.P = P
        self.P1 = P1
        self.K = K
        self.X = X
        self.X1 = X1

        self.X_all = [X]
        self.X1_all = [X1]

    def predict(self):
        #        print('-----------------------------------------------------------')

        self.X1 = np.dot(self.A, self.X_all[-1]) + np.dot(self.B, self.U)
        #        print('X1:\n', self.X1)
        self.P1 = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q
        #        print('P1:\n',self.P1)
        self.X1_all.append(self.X1)
        return (self.X1, self.P1)

    def update(self, Z):  # 输入观测状态向量[x,0,y,0]
        self.K = np.dot(np.dot(self.P1, self.H.T), np.linalg.pinv(np.dot(np.dot(self.H, self.P1), self.H.T) + self.R))
        #        print('K:\n', self.K)
        #        print(np.dot(self.P1, self.H.T))
        #        print('*')
        #        print(np.dot(self.H, self.P1))
        #        print(np.dot(np.dot(self.H, self.P1), self.H.T))
        #        print(np.dot(np.dot(self.H, self.P1), self.H.T) + self.R)
        #        print(np.linalg.pinv(np.dot(np.dot(self.H, self.P1), self.H.T) + self.R))

        self.X = self.X1 + np.dot(self.K, Z - np.dot(self.H, self.X1))
        #        print('X:\n', self.X)
        self.P = np.dot(np.eye(self.K.shape[0]) - np.dot(self.K, self.H), self.P1)
        #        print('P:\n', self.P)
        self.X_all.append(self.X)
        #        print('-----------------------------------------------------------')
        return (self.X, self.P, self.K)

    def getX(self):
        return self.X_all

    def getcount(self):
        return KFT.count
