import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def gaussian_filter(image, alpha=1, plot_flag=False):
    """ Gaussian filter is a linear filter that uses a Gaussian kernel 
    to smooth and reduce noise in an image.

    Args:
        - image: opencv image
        - alpha: standard deviation for the Gaussian kernel.
        - plot_flag: if True, plots original and filtered images.
    Returns:
        - filtered_image: filtered image.
    """
    kernel_size = 2 * int(3 * alpha) + 1
    kernel = cv2.getGaussianKernel(kernel_size, alpha)
    gaussian_kernel = kernel @ kernel.T
    filtered_image = cv2.filter2D(image, -1, gaussian_kernel)

    if plot_flag:
        cmap = 'gray' if len(image.shape) == 2 else None
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(image, cmap=cmap)
        axs[0].set_title("Original Image")
        axs[0].axis('off')
        axs[1].imshow(filtered_image, cmap=cmap)
        axs[1].set_title(f'Gaussian Filter (alpha={alpha})')
        axs[1].axis('off')
        plt.tight_layout()
        plt.show()

    return filtered_image


def study_gaussian_filter(image, alpha_values=[1, 2, 3], plot_flag=False):
    """ Compares the effect of the Gaussian filter for multiple alphas. """
    images = {alpha: gaussian_filter(image, alpha=alpha, plot_flag=False)
              for alpha in alpha_values}

    if plot_flag:
        n = len(alpha_values) + 1
        rows = math.ceil(math.sqrt(n))
        cols = math.ceil(n / rows)
        fig, axs = plt.subplots(rows, cols, figsize=(12, 8))
        axs = axs.flatten()
        cmap = 'gray' if len(image.shape) == 2 else None

        axs[0].imshow(image, cmap=cmap)
        axs[0].set_title("Original Image")
        axs[0].axis('off')

        for i, (alpha, filtered) in enumerate(images.items(), start=1):
            axs[i].imshow(filtered, cmap=cmap)
            axs[i].set_title(f'Gaussian Filter (α={alpha})')
            axs[i].axis('off')

        for j in range(i + 1, len(axs)):
            axs[j].axis('off')

        plt.tight_layout()
        plt.show()

    return images