import cv2 
import math
import numpy as np
import matplotlib.pyplot as plt


def clahe_histogram(image, clip_limit=2.0, tile_grid_size=(8, 8), plot_flag=False):
    """Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to a grayscale image.
    Args:
        image (numpy.ndarray): Input grayscale image.
        clip_limit (float, optional): Threshold for contrast limiting. Defaults to 2.0.
        tile_grid_size (tuple, optional): Size of grid for histogram equalization. Defaults to (8, 8).
    Returns:
        numpy.ndarray: Image after applying CLAHE.
    """
    if len(image.shape) != 2:
        raise ValueError("Input image must be a grayscale image.")
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    clahe_image = clahe.apply(image)
    clahe_hist = cv2.calcHist([clahe_image], [0], None, [256], [0, 256])
    hist = cv2.calcHist([image], [0], None, [256], [0, 256]) 
    if plot_flag:
        fig, axs = plt.subplots(2, 2, figsize=(12, 12), sharey='row')
        axs[0, 0].set_title('Original Gray Scale Image')
        axs[0, 0].imshow(image, cmap='gray')
        axs[0, 0].axis('off')
        axs[0, 1].set_title('CLAHE Gray Scale Image')
        axs[0, 1].imshow(clahe_image, cmap='gray')
        axs[0, 1].axis('off')
        axs[1, 0].set_title('Histogram of Original Image')
        axs[1, 0].set_xlabel('Pixel Intensity')
        axs[1, 0].set_ylabel('Number of Pixels')
        axs[1, 0].bar(range(256), hist.flatten(), width=1.0)
        axs[1, 0].set_xlim([0, 256])
        axs[1, 1].set_title('Histogram of CLAHE Image')
        axs[1, 1].set_xlabel('Pixel Intensity')
        axs[1, 1].set_ylabel('Number of Pixels')
        axs[1, 1].bar(range(256), clahe_hist.flatten(), width=1.0)
        axs[1, 1].set_xlim([0, 256])
        plt.show()    
    return clahe_image

def study_clahe_parameters(img_gray, tile_sizes, clip_limits):
    """Study the effect of different CLAHE parameters on a grayscale image.
    Args:
        img_gray (numpy.ndarray): Input grayscale image.
        tile_sizes (list): List of tile sizes to test.
        clip_limits (list): List of clip limits to test.
    Returns:
        None: Displays a grid of images with different CLAHE parameters.
    """

    fig, axs = plt.subplots(len(clip_limits), len(tile_sizes), figsize=(12,12))
    images = {}
    for i, tile_size in enumerate(tile_sizes):
        for j, clip_limit in enumerate(clip_limits):
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
            clahe_image = clahe.apply(img_gray)
            images[(clip_limit, tile_size)] = clahe_image
            axs[j, i].set_title(f'Clip Limit: {clip_limit}, Tile Size: {tile_size}x{tile_size}')
            axs[j, i].imshow(clahe_image, cmap='gray')
            axs[j, i].axis('off')
    plt.tight_layout()
    plt.show()
    return images
