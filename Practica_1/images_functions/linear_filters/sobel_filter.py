import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def sobel_filter(image, kernel_size=3, ddepth=cv2.CV_16S, scale=1, delta=0, plot_flag=False):
    """  https://docs.opencv.org/3.4/d2/d2c/tutorial_sobel_derivatives.html
    Sober filter is used for edge detection by approximating the gradient of the image intensity.
    Useful for edge detection in horizontal and vertical directions.
    """
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd.")
    grad_x = cv2.Sobel(image, ddepth, 1, 0, ksize=kernel_size, scale=scale, delta=delta)
    grad_y = cv2.Sobel(image, ddepth, 0, 1, ksize=kernel_size, scale=scale, delta=delta)
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
        axs[1].set_title(f'Sobel Filter (ksize={kernel_size})')
        axs[1].axis('off')
        plt.tight_layout()
        plt.show()

    return combined


def study_sobel_filter(image, kernel_sizes=[3, 5, 7], ddepth=cv2.CV_16S, scale=1, delta=0, plot_flag=False):
    """ Compares Sobel filter results for multiple kernel sizes. """
    images = {size: sobel_filter(image, kernel_size=size, ddepth=ddepth, scale=scale, delta=delta, plot_flag=False)
              for size in kernel_sizes}

    if plot_flag:
        n = len(kernel_sizes) + 1
        rows = math.ceil(math.sqrt(n))
        cols = math.ceil(n / rows)
        fig, axs = plt.subplots(rows, cols, figsize=(12, 8))
        axs = axs.flatten()
        cmap = 'gray' if len(image.shape) == 2 else None

        axs[0].imshow(image, cmap=cmap)
        axs[0].set_title("Original Image")
        axs[0].axis('off')

        for i, (size, filtered) in enumerate(images.items(), start=1):
            axs[i].imshow(filtered, cmap=cmap)
            axs[i].set_title(f'Sobel Filter (ksize={size})')
            axs[i].axis('off')

        for j in range(i + 1, len(axs)):
            axs[j].axis('off')

        plt.tight_layout()
        plt.show()

    return images
