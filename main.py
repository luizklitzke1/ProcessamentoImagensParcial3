#Alunos: Arthur B Pinotti, Kaue Rebli,, Luiz G Klitzke
#Doc de base do OpenCV: https://docs.opencv.org/3.4/d1/d5c/tutorial_py_kmeans_opencv.html

import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    path_datasets = "datasets"
    file_names = os.listdir(path_datasets)

    for file_name in file_names:
        img_path = os.path.join(path_datasets, file_name)

        img = cv.imread(img_path)
        plt.title("1- " + file_name)
        plt.imshow(img)
        plt.show()

        Z = img.reshape((-1,3))
        Z = np.float32(Z)

        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 5
        ret, label, center = cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))
        plt.title("2- K-Means (" + str(K) + ")")
        plt.imshow(res2)
        plt.show()

        plt.title("3- Grayscale")
        img_grayscale = cv.cvtColor(res2, cv.COLOR_BGR2GRAY)
        plt.imshow(img_grayscale, cmap = "gray")
        plt.show()
            
        #Base em https://pyimagesearch.com/2021/04/28/opencv-image-histograms-cv2-calchist/
        hist = cv.calcHist([img_grayscale], [0], None, [256], [0, 256])
        plt.title('4- Histograma')

        #Pega a média do primeiro (mais escuro) tom de cinza no histograma - que representa os núcleos - para separar esse tom
        colors_hist = np.nonzero(hist.flatten())[0]
        first_gray = colors_hist[1]
        second_gray = colors_hist[2]
        lower_threshold = first_gray / 2
        upper_threshold = (first_gray + second_gray) / 2
        plt.axvline(x = lower_threshold, color = "g", label = "lower_threshold")
        plt.axvline(x = upper_threshold, color = "r", label = "upper_threshold")
        plt.legend()
        plt.plot(hist)
        plt.show()

        plt.title("5 - Threshold")
        thresholded_img = cv.inRange(img_grayscale, lower_threshold, upper_threshold)
        plt.imshow(thresholded_img, cmap = "gray")
        plt.show()
    
        contours, hierarchy = cv.findContours(thresholded_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        img_contours = cv.drawContours(img, contours, -1, (0, 255, 0), 1)
        plt.title("6 - Núcleos: " + str(len(contours)))
        plt.imshow(img_contours)
        plt.show()