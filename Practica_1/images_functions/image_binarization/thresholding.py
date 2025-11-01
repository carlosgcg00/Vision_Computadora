import matplotlib.pyplot as plt
import cv2
import numpy as np

def img_binarization(
    image,
    threshold=127,
    max_value=255,
    use_otsu=False,
    plot_flag=False,
    image_title='Original Image'
):
    """
    Binarize an image using a fixed or Otsu thresholding method.
    
    Args:
        - image: Input image (grayscale or BGR).
        - threshold: Threshold value used for fixed thresholding (ignored if use_otsu=True).
        - max_value: Maximum value used with the THRESH_BINARY type (default is 255).
        - use_otsu: If True, apply Otsu's method automatically (threshold is forced to 0).
        - plot_flag: If True, display the original and binarized images using matplotlib.
        - image_title: Title for the original image in the plot.
    
    Returns:
        - binary_image: The binarized image (uint8).
    """

    # Ensure grayscale uint8
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype(np.uint8)

    # Select thresholding mode
    if use_otsu:
        t = 0
        thresh_type = cv2.THRESH_BINARY + cv2.THRESH_OTSU
    else:
        t = int(threshold)
        thresh_type = cv2.THRESH_BINARY

    # Apply thresholding
    _, binary_image = cv2.threshold(image, t, max_value, thresh_type)

    # Optional plotting
    if plot_flag:
        fig, axs = plt.subplots(1, 2, figsize=(8, 10))
        axs[0].set_title(image_title)
        axs[0].imshow(image, cmap='gray')
        axs[0].axis('off')

        title = 'Otsu Binarization' if use_otsu else f'Fixed Threshold (t={t})'
        axs[1].set_title(title)
        axs[1].imshow(binary_image, cmap='gray')
        axs[1].axis('off')

        plt.tight_layout()
        plt.show()

    return binary_image


def study_img_binarization(image, threshold=127, max_value=255, plot_flag = False, image_title = 'Original Image'):
    """
    Binarize an image using a specified thresholding method.
    Args:
        - image: Input image (grayscale).
        - threshold: Threshold value (default is 127).
        - max_value: Maximum value to use with the THRESH_BINARY and THRESH_BINARY_INV thresholding types (default is 255).
    Returns:
        - binary_image: Binarized image.
    """
    # Apply thresholding
    _, binary_image = cv2.threshold(image, threshold, max_value, cv2.THRESH_BINARY)
    th_o, binary_image_otsu = cv2.threshold(image, 0, max_value, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    
    if plot_flag:
        # Plot original and binarized images
        fig, axs = plt.subplots(1, 3, figsize=(8, 10))
        axs[0].set_title(image_title)
        axs[0].imshow(image, cmap='gray')
        axs[0].axis('off')

        title ='Binarized Image'
        axs[1].set_title(title)
        axs[1].imshow(binary_image, cmap='gray')
        axs[1].axis('off')
        
        title =f'Otsu Binarized Image\nThreshold: {th_o:2f}'
        axs[2].set_title(title)
        axs[2].imshow(binary_image, cmap='gray')
        axs[2].axis('off')

        plt.tight_layout()
        plt.show()

    return binary_image, binary_image_otsu


def img_adaptive_binarization(
    image,
    max_value=255,
    threshold_type=cv2.THRESH_BINARY,
    block_size=11,
    C=2,
    method='mean',
    plot_flag=False,
    image_title='Original Image'
):
    """
    Adaptive mean and Gaussian thresholding.
    """
    # Ensure grayscale uint8
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype(np.uint8)

    # Choose adaptive method
    if method.lower() == 'mean':
        adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C
    elif method.lower() == 'gaussian':
        adaptive_method = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
    else:
        raise ValueError("Invalid method. Choose 'mean' or 'gaussian'.")

    # Apply adaptive thresholding
    binary_image = cv2.adaptiveThreshold(
        image, max_value, adaptive_method, threshold_type, block_size, C
    )

    # Optional plotting
    if plot_flag:
        fig, axs = plt.subplots(1, 2, figsize=(8, 10))
        axs[0].set_title(image_title)
        axs[0].imshow(image, cmap='gray')
        axs[0].axis('off')

        axs[1].set_title(f'Adaptive {method.capitalize()} Thresholding')
        axs[1].imshow(binary_image, cmap='gray')
        axs[1].axis('off')

        plt.tight_layout()
        plt.show()

    return binary_image


def study_img_adaptive_binarization(
    image,
    max_value=255, 
    threshold_type=cv2.THRESH_BINARY,
    block_size=11,
    C=2,
    plot_flag=False,
    image_title='Original Image'
):
    """
    Binarize an image using adaptive thresholding for multiple block sizes.

    Args:
        - image: Input image (grayscale).
        - max_value: Maximum value for THRESH_BINARY/INV.
        - threshold_type: Thresholding type (default is cv2.THRESH_BINARY).
        - block_size: Single integer or list of block sizes (must be odd and >1).
        - C: Constant subtracted from the mean or weighted mean.
        - plot_flag: If True, display a grid of results.
        - image_title: Title for the original image.

    Returns:
        - results_mean: Dict mapping block size → binary image (mean).
        - results_gaussian: Dict mapping block size → binary image (gaussian).
    """

    # Ensure grayscale uint8
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.astype(np.uint8)

    # Convert single block size to list
    if isinstance(block_size, int):
        block_sizes = [block_size]
    else:
        block_sizes = block_size

    results_mean = {}
    results_gaussian = {}

    # Compute all adaptive thresholding results
    for bs in block_sizes:
        if bs % 2 == 0 or bs <= 1:
            raise ValueError(f"Block size must be odd and >1, got {bs}")
        binary_image_mean = cv2.adaptiveThreshold(
            image, max_value, cv2.ADAPTIVE_THRESH_MEAN_C,
            threshold_type, bs, C
        )
        binary_image_gaussian = cv2.adaptiveThreshold(
            image, max_value, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            threshold_type, bs, C
        )
        results_mean[bs] = binary_image_mean
        results_gaussian[bs] = binary_image_gaussian

    # Plot results grid
    if plot_flag:
        n_cols = len(block_sizes) + 1  # +1 for original
        n_rows = 2  # Mean (fila 1), Gaussian (fila 2)
        fig_width = 4 * n_cols
        fig_height = 4 * n_rows
        fig, axs = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))
        axs = np.atleast_2d(axs)  # Ensure 2D array

        # Primera columna: imagen original
        for r in range(n_rows):
            axs[r, 0].imshow(image, cmap='gray')
            if r == 0:
                axs[r, 0].set_title(image_title)
            else:
                axs[r, 0].set_title("")
            axs[r, 0].axis('off')

        # Fila 1: Adaptive Mean
        for i, bs in enumerate(block_sizes, start=1):
            axs[0, i].imshow(results_mean[bs], cmap='gray')
            axs[0, i].set_title(f'Adaptive Mean\nBlock={bs}')
            axs[0, i].axis('off')

        # Fila 2: Adaptive Gaussian
        for i, bs in enumerate(block_sizes, start=1):
            axs[1, i].imshow(results_gaussian[bs], cmap='gray')
            axs[1, i].set_title(f'Adaptive Gaussian\nBlock={bs}')
            axs[1, i].axis('off')

        plt.tight_layout()
        plt.show()

    return results_mean, results_gaussian