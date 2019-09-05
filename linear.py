import numpy as np
import matplotlib.pyplot as plt

#1、创建数据集
def createdata():
    np.random.seed(12)
    num_observations = 500

    x1 = np.random.multivariate_normal([0, 0], [[1, .75], [.75, 1]], num_observations)
    x2 = np.random.multivariate_normal([1, 4], [[1, .75], [.75, 1]], num_observations)

    X = np.vstack((x1, x2)).astype(np.float32)
    a = np.zeros(num_observations) - [1]
    Y = np.hstack((a, np.ones(num_observations)))
    return X,Y

#封装好的感知机模型
class Perceptron:
    def __init__(self,x,y,a=0.1, iter = 100):
        self.x=x
        self.y=y
        self.w=np.zeros((x.shape[1],1))   #初始化权重，w1,w2均为0
        self.b=0
        self.a=a  #学习率，默认0.1
        self.iter = iter # 迭代次数，默认100次
        self.numsamples=self.x.shape[0] # 样本数
        self.numfeatures=self.x.shape[1] # 特征数
        self.errors = []

    def sign(self,w,b,x):
        y=np.dot(x,w)+b
        return int(y)

    # 梯度下降法
    def update(self,label_i,data_i):
        tmp=label_i*self.a*data_i
        tmp=tmp.reshape(self.w.shape)
        #更新w和b
        self.w=tmp+self.w
        self.b=self.b+label_i*self.a

    # 训练函数
    def train(self):
            for k in range(self.iter):
                count=0
                for i in range(self.numsamples):
                    tmpY=self.sign(self.w,self.b,self.x[i,:])
                    if tmpY*self.y[i]<=0:#如果是一个误分类实例点
                        count += 1
                        self.update(self.y[i],self.x[i,:])
                self.errors.append(float(count)/float(self.numsamples))
            print('最终训练得到的w和b为：',self.w,self.b)
            return self.w,self.b

    # 每轮迭代时误差率
    def calc_error(self):
        count=0
        for i in range(self.numsamples):
            tmpY=self.sign(self.w, self.b, self.x[i,:])
            if tmpY*self.y[i]<0:
                count += 1
        self.errors.append(float(count)/float(self.numsamples))

#画图描绘
class Picture:
    def __init__(self,data,labels,w,b):
        self.b=b
        self.w=w
        plt.figure(1)
        plt.title('Perceptron Learning Algorithm',size=14)
        plt.xlabel('x0-axis',size=14)
        plt.ylabel('x1-axis',size=14)

        xData=np.linspace(-5,5,100)
        yData=self.expression(xData)
        plt.plot(xData,yData,color='r',label='sample data')

        for i in range(data.shape[0]):
            if labels[i] < 0:
                plt.scatter(data[i][0], data[i][1], s=50)
            elif labels[i] > 0:
                plt.scatter(data[i][0], data[i][1], s=50, marker='x')
            # plt.savefig('2d.png',dpi=75) # 储存图片

    def expression(self,x):
        y=(-self.b-self.w[0]*x)/self.w[1] #注意在此，把x0，x1当做两个坐标轴，把x1当做自变量，x2为因变量
        return y

    def Show(self):
        plt.show()


if __name__ == '__main__':
    samples,labels=createdata() # 得到样本标签
    # myperceptron=Perceptron(x=samples,y=labels,a=0.1,iter=50) # 实例化分类器模型
    myperceptron=Perceptron(x=samples,y=labels,a=0.1,iter=100) # 实例化分类器模型
    weights,bias=myperceptron.train()   # 训练
    Picture=Picture(samples,labels,weights,bias) # 绘图分界线
    Picture.Show()

    # 绘制随迭代次数增加的误差曲线
    num = [i for i in range(len(myperceptron.errors))]
    plt.figure()
    plt.plot(num, myperceptron.errors, color='r', label='error curve')
    plt.show()

