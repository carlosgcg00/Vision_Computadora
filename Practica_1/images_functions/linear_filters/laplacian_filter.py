import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def laplacian_filter(image, kernel_size=3, ddepth=cv2.CV_16S, plot_flag=False):
    """ https://docs.opencv.org/3.4/d5/db5/tutorial_laplace_operator.html
    Laplacian filter is a second derivative filter used for edge detection. It highlights regions of rapid intensity change.
    """
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd.")
    laplacian = cv2.Laplacian(image, ddepth, ksize=kernel_size)
    abs_laplacian = cv2.convertScaleAbs(laplacian)

    if plot_flag:
        cmap = 'gray' if len(image.shape) == 2 else None
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(image, cmap=cmap)
        axs[0].set_title("Original Image")
        axs[0].axis('off')
        axs[1].imshow(abs_laplacian, cmap=cmap)
        axs[1].set_title(f'Laplacian Filter (ksize={kernel_size})')
        axs[1].axis('off')
        plt.tight_layout()
        plt.show()

    return abs_laplacian


def study_laplacian_filter(image, kernel_sizes=[3, 5, 7], ddepth=cv2.CV_16S, plot_flag=False):
    """ Compares Laplacian filter for multiple kernel sizes. """
    images = {size: laplacian_filter(image, kernel_size=size, ddepth=ddepth, plot_flag=False)
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
            axs[i].set_title(f'Laplacian Filter (ksize={size})')
            axs[i].axis('off')

        for j in range(i + 1, len(axs)):
            axs[j].axis('off')

        plt.tight_layout()
        plt.show()

    return images
