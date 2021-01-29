import cv2 
import numpy as np
import math
from scipy import ndimage
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import os


def test_denoise(path_to_test):
    ls_path_image = os.listdir(path_to_test)  
    ls_label = [int(i.split("_")[-1].replace(".jpg", "")) for i in ls_path_image]     
    ls_image_gray = [cv2.imread(path_to_test + i, 0) for i in ls_path_image]

    ls_gau = [cv2.GaussianBlur(img_gray,(7,7), 0) for img_gray in ls_image_gray]
    ls_average = [cv2.blur(img_gray, (5,5)) for img_gray in ls_image_gray]
    ls_median = [cv2.medianBlur(img_gray,5) for img_gray in ls_image_gray]
    ls_bilateral = [cv2.bilateralFilter(img_gray, 5, 75, 75, ) for img_gray in ls_image_gray]

    return ls_gau, ls_average, ls_median, ls_bilateral, ls_label

#   5-10-15-20-25


def test_thresh(ls_img, thresh):
    size_thresh_min = 1
    size_thresh_max = 200
    ls_predict = []
    ls_img_thresh = []
    for img in ls_img:
        _, img_thresh = cv2.threshold(img,thresh,255,cv2.THRESH_BINARY_INV)
        n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img_thresh)
        num_box = 0
        for i in range(1, n_labels):
            if stats[i, cv2.CC_STAT_AREA] > size_thresh_min and stats[i, cv2.CC_STAT_AREA] <= size_thresh_max:
                num_box+=1
        ls_predict.append(num_box)
        ls_img_thresh.append(img_thresh)

    # fig = plt.figure(figsize=(1,11))
    # columns = 11
    # rows = 1
    # for i in range(1, columns*rows +1):
    #     fig.add_subplot(rows, columns, i)
    #     plt.imshow(ls_img_thresh[i-1])
    # plt.show()
    return ls_predict

def hien_RMSE(true, predict):
    return np.sqrt(np.square(np.subtract(true,predict)).mean())

if __name__ == "__main__":
    path_to_test = "C:/Users/hiennt141/Desktop/project_VBDI/project_CV/data/ruler/"
    ls_gau, ls_average, ls_median, ls_bilateral, ls_label = test_denoise(path_to_test)
    print(len(ls_label))
    print(ls_label)
    print()
    for i in range(5,35, 5):
        print("Thresh:", i)
        print(hien_RMSE(ls_label, test_thresh(ls_gau, i)))
        print(hien_RMSE(ls_label, test_thresh(ls_average, i)))
        print(hien_RMSE(ls_label, test_thresh(ls_median, i)))
        print(hien_RMSE(ls_label, test_thresh(ls_bilateral, i)))
        print("####################")
'''