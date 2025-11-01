import cv2 
import math
import numpy as np
import matplotlib.pyplot as plt

def equalize_histogram(image, plot_flag=False):
    """Equalize the histogram of a grayscale image.
    Args:
        image (numpy.ndarray): Input grayscale image.
    Returns:
        numpy.ndarray: Image with equalized histogram.
    """
    if len(image.shape) != 2:
        raise ValueError("Input image must be a grayscale image.")
    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    equalized_image = cv2.equalizeHist(image)
    equalized_hist = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])
    cdf_eq = equalized_hist.cumsum()
    cdf_normalized_eq = cdf_eq * float(equalized_hist.max()) / cdf_eq.max()
    if plot_flag:
        fig, axs = plt.subplots(2, 2, figsize=(12, 12))
        axs[0, 0].set_title('Original Gray Scale Image')
        axs[0, 0].imshow(image, cmap='gray')
        axs[0, 0].axis('off')
        axs[0, 1].set_title('Equalized Gray Scale Image')
        axs[0, 1].imshow(equalized_image, cmap='gray')
        axs[0, 1].axis('off')
        axs[1, 0].set_title('Histogram of Original Image')
        axs[1, 0].set_xlabel('Pixel Intensity')
        axs[1, 0].set_ylabel('Number of Pixels')
        axs[1, 0].bar(range(256), hist.flatten(), width=1.0, label='Histogram')
        axs[1, 0].plot(cdf_normalized, color = 'red', label='cdf')
        axs[1, 0].set_xlim([0, 256])
        axs[1, 1].set_title('Histogram of Equalized Image')
        axs[1, 1].set_xlabel('Pixel Intensity')
        axs[1, 1].set_ylabel('Number of Pixels')
        axs[1, 1].bar(range(256), equalized_hist.flatten(), width=1.0, label='Histogram')
        axs[1, 1].plot(cdf_normalized_eq, color = 'red', label='cdf')
        axs[1, 1].set_xlim([0, 256])
        plt.legend()
        plt.show()
    return equalized_image
