import torch
import cv2
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(1, 10, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(10, 20, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )
        self.fc = torch.nn.Sequential(
            torch.nn.Linear(320, 50),
            torch.nn.Linear(50, 10),
        )

    def forward(self, x):
        batch_size = x.size(0)
        x = self.conv1(x)  # 一层卷积层,一层池化层,一层激活层(图是先卷积后激活再池化，差别不大)
        x = self.conv2(x)  # again
        x = x.view(batch_size, -1)  # flatten 变成全连接网络需要的输入 (batch, 20,4,4) ==> (batch,320), -1 此处自动算出的是320
        x = self.fc(x)
        return x  # 输出维度为10，即0~9


if __name__ == '__main__':
    device = torch.device('cuda')
    model = torch.load('Model/Mnist.pth')  # 加载模型
    model = model.to(device)
    model.eval()  # 切换模型为测试模式

    image = cv2.imread('Image/x.png')  # 读取图片

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 灰度处理
    _, img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)  # 二值化

    img = cv2.resize(img, (28, 28))  # 调整大小为28x28

    img = np.array(img).astype(np.float32)
    img = np.expand_dims(img, 0)
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img)
    img = img.to(device)
    output = model(Variable(img))
    prob = F.softmax(output, dim=1)
    prob = Variable(prob)
    prob = prob.cpu().numpy()  # 用GPU的数据训练的模型保存的参数都是gpu形式的，要显示则先要转回cpu，再转回numpy模式
    print(prob)  # prob是10个分类的概率
    pred = np.argmax(prob)  # 选出概率最大的一个
    print(pred.item())  # 输出结果
