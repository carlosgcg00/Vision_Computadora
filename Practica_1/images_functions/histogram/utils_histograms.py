import cv2 
import math
import numpy as np
import matplotlib.pyplot as plt



def plot_histogram(image):
    """Plot the histogram of an image on grey scale or color.
    Args:
        image (numpy.ndarray): Input image.
    Returns:
        subplots: (2,1) original image and histogram
    """
    if len(image.shape) == 2:
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        cdf = hist.cumsum()
        cdf_normalized = cdf * float(hist.max()) / cdf.max()
        cmap = 'gray'
        title = 'Gray Scale Image'
        fig, axs = plt.subplots(2, 1, figsize=(6,6))
        axs[0].set_title(title)
        axs[0].imshow(image, cmap='gray')
        axs[0].axis('off')
        # Plot histogram
        axs[1].set_title('Image Histogram')
        axs[1].set_xlabel('Pixel Intesity')
        axs[1].set_ylabel('Number of pixels')
        axs[1].bar(range(256), hist.flatten(), width=1.0, label ='Histogram')
        axs[1].plot(cdf_normalized, color = 'red', label='cdf')
        axs[1].set_xlim([0, 256])
        plt.show()
    else:
        hist_r = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([image], [2], None, [256], [0, 256])
        cdf_r = hist_r.cumsum()
        cdf_g = hist_g.cumsum()
        cdf_b = hist_b.cumsum()
        cdf_normalized_r = cdf_r * float(hist_r.max()) / cdf_r.max()
        cdf_normalized_g = cdf_g * float(hist_g.max()) / cdf_g.max()
        cdf_normalized_b = cdf_b * float(hist_b.max()) / cdf_b.max()
        cmap = None
        title = 'Color Image'
        fig, axs = plt.subplots(2, 1, figsize=(6,6))
        axs[0].set_title(title)
        axs[0].imshow(image)
        axs[0].axis('off')
        # Plot histogram
        axs[1].set_title('Image Histogram')
        axs[1].set_xlabel('Pixel Intesity')
        axs[1].set_ylabel('Number of pixels')
        axs[1].bar(range(256), hist_r.flatten(), color='r', width=1.0, label='Red channel')
        axs[1].bar(range(256), hist_g.flatten(), color='g', width=1.0, label='Green channel')
        axs[1].bar(range(256), hist_b.flatten(), color='b', width=1.0, label='Blue channel')
        axs[1].plot(cdf_normalized_r, color = 'darkred', label='cdf Red')
        axs[1].plot(cdf_normalized_g, color = 'darkgreen', label='cdf Green')
        axs[1].plot(cdf_normalized_b, color = 'darkblue', label='cdf Blue')
        axs[1].set_xlim([0, 256])
        axs[1].legend()
        plt.show()
        

def plot_img_histogram_for_subplot(img, ax_img, ax_hist, title):
    """Plot image and histogram in given axes for subplot.
    Args:
        img (numpy.ndarray): Input image.
        ax_img (matplotlib.axes.Axes): Axes for the image.
        ax_hist (matplotlib.axes.Axes): Axes for the histogram.
        title (str): Title for the image plot.
    Returns:
        None: Plots the image and histogram in the given axes.
    """
    ax_img.imshow(img, cmap='gray')
    ax_img.set_title(title)
    ax_img.axis('off')
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * float(hist.max()) / cdf.max()
    ax_hist.bar(range(256), hist.flatten(), width=1.0, label='Histogram')
    ax_hist.plot(cdf_normalized, color = 'red', label='cdf')
    ax_hist.set_xlim([0, 255])
    ax_hist.set_yticks([])    


def compare_and_plot_hist_transformations(images, titles_images):
    """
    Compare and plot multiple images and their histograms.
    Args:
        images (list): List of images to be compared.
        titles_images (list): List of titles for each image.
    Return:
        Subplot grid with images and histograms.
    """
    num_images = len(images)
    max_cols = 4  # máximo número de columnas de imágenes
    cols = min(num_images, max_cols)
    rows = 2 * math.ceil(num_images / max_cols)  # dos filas por cada bloque de imágenes

    fig, axs = plt.subplots(rows, cols, figsize=(4*cols, 3*rows))

    # Si solo hay una fila o una columna, garantizamos que axs sea indexable como matriz 2D
    axs = np.atleast_2d(axs)

    for idx, (image, title) in enumerate(zip(images, titles_images)):
        row_block = (idx // max_cols) * 2      # bloque de filas (cada bloque tiene 2 filas)
        col = idx % max_cols                   # columna dentro del bloque
        plot_img_histogram_for_subplot(
            img=image,
            ax_img=axs[row_block, col],
            ax_hist=axs[row_block + 1, col],
            title=title
        )

    # Ocultar ejes vacíos si el número de imágenes no llena todas las columnas
    total_plots = cols * (rows // 2)
    for j in range(num_images, total_plots):
        row_block = (j // max_cols) * 2
        col = j % max_cols
        axs[row_block, col].axis('off')
        axs[row_block + 1, col].axis('off')

    plt.tight_layout()
    plt.show()