import os
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    path_datasets = "imgs"
    file_names = os.listdir(path_datasets)

    for file_name in file_names:
        fig, axs = plt.subplots(1, 6)
        fig.set_figheight(5)
        fig.set_figwidth(30)
        img_path = os.path.join(path_datasets, file_name)

        img = cv.imread(img_path)
        axs[0].set_title("1- " + file_name)
        axs[0].imshow(img)

        Z = img.reshape((-1,3))
        Z = np.float32(Z)

        criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 10, 1.0)
        K = 4
        ret, label, center = cv.kmeans(Z, K, None, criteria, 10, cv.KMEANS_RANDOM_CENTERS)

        center = np.uint8(center)
        res = center[label.flatten()]
        res2 = res.reshape((img.shape))
        axs[1].set_title("2- K-Means (" + str(K) + ")")
        axs[1].imshow(res2)

        axs[2].set_title("3- Grayscale")
        img_grayscale = cv.cvtColor(res2, cv.COLOR_BGR2GRAY)
        axs[2].imshow(img_grayscale, cmap = "gray")
        
        hist = cv.calcHist([img_grayscale], [0], None, [256], [0, 256])
        colors_hist = np.nonzero(hist.flatten())[0]
        first_gray = colors_hist[1]
        second_gray = colors_hist[2]
        lower_threshold = first_gray / 2
        upper_threshold = (first_gray + second_gray) / 2

        axs[3].set_title('4- Histograma')
        axs[3].axvline(x = lower_threshold, color = "g", label = "lower_threshold")
        axs[3].axvline(x = upper_threshold, color = "r", label = "upper_threshold")
        axs[3].legend()
        axs[3].plot(hist)

        thresholded_img = cv.inRange(img_grayscale, lower_threshold, upper_threshold)
        axs[4].set_title("5 - Threshold")
        axs[4].imshow(thresholded_img, cmap = "gray")
        
        contours, hierarchy = cv.findContours(thresholded_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        img_contours = cv.drawContours(img, contours, -1, (0, 255, 0), 1)
        axs[5].set_title("6 - Núcleos: " + str(len(contours)))
        axs[5].imshow(img_contours)

        plt.tight_layout(pad = 0.5, w_pad = 0.5, h_pad = 2)