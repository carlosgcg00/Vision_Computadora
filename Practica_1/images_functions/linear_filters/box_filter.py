import cv2
import numpy as np
import matplotlib.pyplot as plt
import math


def box_filter(image, kernel_size=3, plot_flag=False):
    """ Box filter applies the 'average filter' to the image. 
    Smooths the image by averaging pixel values in a local neighborhood.
    Used to reduce noise in the image.

    Args:
        - image: opencv image
        - kernel_size: size of the matrix of ones used.
        - plot_flag: if True, plots original and filtered images.
    Returns: 
        - filtered_image: filtered image.
    """
    kernel = 1 / (kernel_size * kernel_size) * np.ones((kernel_size, kernel_size))
    filtered_image = cv2.filter2D(image, -1, kernel)

    if plot_flag:
        cmap = 'gray' if len(image.shape) == 2 else None
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(image, cmap=cmap)
        axs[0].set_title("Original Image")
        axs[0].axis('off')
        axs[1].imshow(filtered_image, cmap=cmap)
        axs[1].set_title(f'Box Filter ({kernel_size}x{kernel_size})')
        axs[1].axis('off')
        plt.tight_layout()
        plt.show()

    return filtered_image


def study_box_filter(image, kernel_sizes=[3, 4, 5], plot_flag=False):
    """ Compares the effect of the box filter for various kernel sizes. """
    images = {size: box_filter(image, kernel_size=size, plot_flag=False) for size in kernel_sizes}

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
            axs[i].set_title(f'Box Filter ({size}x{size})')
            axs[i].axis('off')

        for j in range(i + 1, len(axs)):
            axs[j].axis('off')

        plt.tight_layout()
        plt.show()

    return images