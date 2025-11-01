import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def scharr_filter(image, ddepth=cv2.CV_16S, scale=1, delta=0, border_type=cv2.BORDER_DEFAULT, plot_flag=False):
    """ 
    https://docs.opencv.org/4.x/d4/d86/group__imgproc__filter.html#gaa13106761eedf14798f37aa2d60404c9
    Scharr filter computes the image gradient like Sobel, but with higher rotational symmetry 
    for a 3x3 kernel. It provides more accurate results for diagonal edges.
     """
    grad_x = cv2.Scharr(image, ddepth, 1, 0, scale=scale, delta=delta, borderType=border_type)
    grad_y = cv2.Scharr(image, ddepth, 0, 1, scale=scale, delta=delta, borderType=border_type)
    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)
    combined = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    if plot_flag:
        cmap = 'gray' if len(image.shape) == 2 else None
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(image, cmap=cmap)
        axs[0].set_title("Original Image")
        axs[0].axis('off')
        axs[1].imshow(combined, cmap=cmap)
        axs[1].set_title("Scharr Filter (3x3)")
        axs[1].axis('off')
        plt.tight_layout()
        plt.show()

    return combined


def study_scharr_filter(image, ddepth=cv2.CV_16S, scale=1, delta=0, border_type=cv2.BORDER_DEFAULT, plot_flag=False):
    """ Compares Scharr filter with the original image (3x3 kernel only). """
    images = {3: scharr_filter(image, ddepth=ddepth, scale=scale, delta=delta,
                               border_type=border_type, plot_flag=False)}

    if plot_flag:
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        cmap = 'gray' if len(image.shape) == 2 else None

        axs[0].imshow(image, cmap=cmap)
        axs[0].set_title("Original Image")
        axs[0].axis('off')

        axs[1].imshow(images[3], cmap=cmap)
        axs[1].set_title("Scharr Filter (3x3)")
        axs[1].axis('off')

        plt.tight_layout()
        plt.show()

    return images
