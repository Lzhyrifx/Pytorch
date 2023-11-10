import time

import torch
import cv2
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np






def model_ocr(x):
    device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    model = torch.load('model.pth')  # 加载模型
    model = model.to(device)
    model.eval()  # 切换模型为测试模式

    # image = cv2.imread(x)  # 读取图片

    cx = cv2.resize(x, (28, 28))

    # 将图像转换为灰度
   # gray_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    threshold = 1

    # 阈值化
    _, img = cv2.threshold(cx, threshold, 255, cv2.THRESH_BINARY)

    cv2.namedWindow("OCR", cv2.WINDOW_NORMAL)  # 创建一个可调整大小的窗口
    cv2.imshow("OCR", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




    time_start = time.time()

    img = np.array(img).astype(np.float32)
    img = np.expand_dims(img, 0)
    img = np.expand_dims(img, 0)
    img = torch.from_numpy(img)
    img = img.to(device)
    output = model(Variable(img))
    prob = F.softmax(output, dim=1)
    prob = Variable(prob)
    prob = prob.cpu().numpy()  # 用GPU的数据训练的模型保存的参数都是gpu形式的，要显示则先要转回cpu，再转回numpy模式
    pred = np.argmax(prob)  # 选出概率最大的一个
    print(pred.item())  # 输出结果

    time_end = time.time()  # 结束计时

    time_c = time_end - time_start  # 运行所花时间
    print('time cost', time_c, 's')