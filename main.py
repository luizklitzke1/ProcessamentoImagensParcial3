#Alunos: Arthur B Pinotti, Kaue Rebli,, Luiz G Klitzke
#Doc de base do OpenCV: https://docs.opencv.org/3.4/d1/d5c/tutorial_py_kmeans_opencv.html

import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    path_datasets = "imgs"
    file_names = os.listdir(path_datasets)

    fig, axs = plt.subplots(len(file_names), 6)
    fig.set_figheight(10)
    fig.set_figwidth(25)

    for i in range(len(file_names)):
        file_name = file_names[i]
        img_path = os.path.join(path_datasets, file_name)

        img = cv.imread(img_path)
        axs[i, 0].set_title("1- " + file_name)
        axs[i, 0].imshow(img)

        Z = img.reshape((-1,3))
        Z = np.float32(Z)

        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 4
        ret, label, center = cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))
        axs[i, 1].set_title("2- K-Means (" + str(K) + ")")
        axs[i, 1].imshow(res2)

        axs[i, 2].set_title("3- Grayscale")
        img_grayscale = cv.cvtColor(res2, cv.COLOR_BGR2GRAY)
        axs[i, 2].imshow(img_grayscale, cmap = "gray")
        
        #Base em https://pyimagesearch.com/2021/04/28/opencv-image-histograms-cv2-calchist/
        hist = cv.calcHist([img_grayscale], [0], None, [256], [0, 256])
        #Pega a média do primeiro (mais escuro) tom de cinza no histograma - que representa os núcleos - para separar esse tom
        colors_hist = np.nonzero(hist.flatten())[0]
        first_gray = colors_hist[1]
        second_gray = colors_hist[2]
        lower_threshold = first_gray / 2
        upper_threshold = (first_gray + second_gray) / 2

        axs[i, 3].set_title('4- Histograma')
        axs[i, 3].axvline(x = lower_threshold, color = "g", label = "lower_threshold")
        axs[i, 3].axvline(x = upper_threshold, color = "r", label = "upper_threshold")
        axs[i, 3].plot(hist)


        thresholded_img = cv.inRange(img_grayscale, lower_threshold, upper_threshold)
        axs[i, 4].set_title("5 - Threshold")
        axs[i, 4].imshow(thresholded_img, cmap = "gray")
        
        contours, hierarchy = cv.findContours(thresholded_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        img_contours = cv.drawContours(img, contours, -1, (0, 255, 0), 1)
        axs[i, 5].set_title("6 - Núcleos: " + str(len(contours)))
        axs[i, 5].imshow(img_contours)

    plt.tight_layout(pad = 0.4, w_pad = 0.5, h_pad = 2)
    plt.show()