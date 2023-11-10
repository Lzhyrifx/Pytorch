import cv2
import numpy as np
from arcaea_offline_ocr import FixRects, crop_xywh
from arcaea_offline_ocr.device.rois.definition import DeviceRoisAutoT2
from arcaea_offline_ocr.device.rois.extractor import DeviceRoisExtractor
from arcaea_offline_ocr.device.rois.masker import DeviceRoisMaskerAutoT2
from arcaea_offline_ocr.ocr import resize_fill_square
from arcaea_offline_ocr.phash_db import ImagePhashDatabase

from test import model_ocr
import torch.nn as nn


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.bn1 = nn.BatchNorm2d(6)
        self.relu1 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(16)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.bn3 = nn.BatchNorm1d(120)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        x = self.relu4(x)
        x = self.fc3(x)
        return x


img_path = "Z:\Project\Python\Pytorch\Arcaea\Screenshot_20231102_170531_3de306e3178de90c8b13f9c4ddd4419a.jpg"
img = cv2.imread(img_path, cv2.IMREAD_COLOR)

rois = DeviceRoisAutoT2(img.shape[1], img.shape[0])
extractor = DeviceRoisExtractor(img, rois)
masker = DeviceRoisMaskerAutoT2()

knn_model = cv2.ml.KNearest.load("digits.knn.dat")
phash_db = ImagePhashDatabase("image-phash-5.1.0.db")

size = 20
roi = masker.far(extractor.score)

cv2.namedWindow("roi", cv2.WINDOW_NORMAL)  # 创建一个可调整大小的窗口
cv2.imshow("roi", roi)
cv2.waitKey(0)
cv2.destroyAllWindows()

contours, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
rects = [cv2.boundingRect(c) for c in contours]
rects = FixRects.connect_broken(rects, roi.shape[1], roi.shape[0])
rects = FixRects.split_connected(roi, rects)
rects = sorted(rects, key=lambda r: r[0])
digit_rois = [resize_fill_square(crop_xywh(roi, rect), size) for rect in rects]
# ocr = DeviceOcr(extractor, masker, knn_model, phash_db)
for i in digit_rois:
    i = cv2.copyMakeBorder(i, 6,3,0,0,cv2.BORDER_CONSTANT)
    cv2.namedWindow("digit rois", cv2.WINDOW_NORMAL)  # 创建一个可调整大小的窗口
    cv2.imshow("digit rois", i)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    ocr = model_ocr(i)
