import cv2
import numpy as np
import sklearn.metrics
from sklearn.datasets import load_breast_cancer
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pylab as plt
import os
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# from knn_utils import getImageData
# from pytorch_classification.Test2_alexnet.knn_utils import getImageData

y_label = ([1, 1, 1, 2, 2, 2])  # 非二进制需要pos_label
y_pre = ([0.3, 0.5, 0.9, 0.8, 0.4, 0.6])


def draw_roc(y_label, y_pre, filename):
    y_label = list(map(int, y_label))
    y_pre = list(map(int, y_pre))
    fpr, tpr, thersholds = roc_curve(y_label, y_pre, pos_label=1)
    f = open("./report/" + filename, "w")
    accuracy = accuracy_score(y_label, y_pre)
    precision = precision_score(y_label, y_pre, average=None)
    recall = recall_score(y_label, y_pre, average=None)
    f1 = f1_score(y_label, y_pre, average=None)
    result = sklearn.metrics.confusion_matrix(y_pre, y_label)
    print(result)
    f.write("accuracy:{},precision:{},recall:{},f1:{}".format(accuracy, precision, recall, f1))
    f.close()
    for i, value in enumerate(thersholds):
        print("%f %f %f" % (fpr[i], tpr[i], value))

    roc_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, 'k--', label='ROC (area = {0:.2f})'.format(roc_auc), lw=2)

    plt.xlim([-0.05, 1.05])  # 设置x、y轴的上下限，以免和边缘重合，更好的观察图像的整体
    plt.ylim([-0.05, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')  # 可以使用中文，但需要导入一些库即字体
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    # plt.imsave("./picture/"+filename)
    plt.savefig("./picture/" + filename)
    plt.show()


def getImageData(directory):
    s = 1
    feature_list = list()
    label_list = list()
    num_classes = 0
    for root, dirs, files in os.walk(directory):
        for d in dirs:
            num_classes += 1
            images = os.listdir(root + d)
            for image in images:
                s += 1
                label_list.append(d)
                feature_list.append(extractFeaturesFromImage(root + d + "/" + image))

    return np.asarray(feature_list), np.asarray(label_list)


def extractFeaturesFromImage(image_file):
    SHAPE = (30, 30)
    img = cv2.imread(image_file)  # 读取图片
    img = cv2.resize(img, SHAPE, interpolation=cv2.INTER_CUBIC)
    # 对图片进行risize操作统一大小
    img = img.flatten()  # 对图像进行降维操作，方便算法计算
    img = img / np.mean(img)  # 归一化，突出特征
    return img


if __name__ == '__main__':
    test_Y = ['normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal']
    y_pre = ['normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'abnormal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'abnormal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'abnormal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'abnormal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'abnormal', 'normal', 'normal', 'normal', 'normal', 'abnormal', 'normal', 'abnormal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'normal', 'abnormal', 'abnormal', 'abnormal', 'normal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'abnormal', 'normal', 'normal', 'abnormal', 'normal', 'normal', 'abnormal', 'normal', 'normal', 'normal', 'normal', 'normal', 'abnormal', 'normal', 'normal', 'abnormal', 'normal', 'normal', 'normal', 'normal', 'abnormal', 'normal', 'abnormal', 'abnormal', 'normal', 'normal', 'normal', 'normal', 'normal', 'abnormal', 'normal', 'normal', 'normal', 'normal', 'normal', 'abnormal', 'normal', 'normal', 'normal', 'normal', 'abnormal', 'normal', 'normal']




    for i in range(len(y_pre)):
        if (y_pre[i] == "normal"):
            y_pre[i] = 1
        else:
            y_pre[i] = 0
    for i in range(len(test_Y)):
        if (test_Y[i] == "normal"):
            test_Y[i] = 1
        else:
            test_Y[i] = 0
    print(y_pre)
    print(test_Y)
    draw_roc(y_label=test_Y, y_pre=y_pre, filename="AlexNet")
