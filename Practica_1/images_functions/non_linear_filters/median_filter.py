import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def median_filter(image, kernel_size=3, plot_flag=False):
    """ Median filter is a non-linear filter that replaces each pixel's value with the median value of its neighborhood.
    The median filter is effective in preserving edges.
    Args:
        - image: opencv image
        - kernel_size: size of the square neighborhood.
        - plot_flag: if True, plots the original and filtered images.
    Returns:
        - image: filtered image.
    """
    filtered_image = cv2.medianBlur(image, kernel_size)

    if plot_flag:
        cmap = 'gray' if len(image.shape) == 2 else None
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(image, cmap=cmap)
        axs[0].set_title("Original Image")
        axs[0].axis('off')

        axs[1].imshow(filtered_image, cmap=cmap)
        axs[1].set_title(f'Median Filter ({kernel_size}x{kernel_size})')
        axs[1].axis('off')

        plt.tight_layout()
        plt.show()

    return filtered_image


def study_median_filter(image, kernel_sizes=[3, 5, 7], plot_flag=False):
    """
    Study the filter o the median for diferent kernels

    Args:
        image (numpy.ndarray): opencv image.
        kernel_sizes (list[int]): kernel sizes.

    Returns:
        dict: set of filtered images for different kernels {kernel: image}
    """
    images = {size: median_filter(image, kernel_size=size, plot_flag=False)
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
            axs[i].set_title(f'Median Filter ({size}x{size})')
            axs[i].axis('off')

        for j in range(i + 1, len(axs)):
            axs[j].axis('off')

        plt.tight_layout()
        plt.show()

    return images
