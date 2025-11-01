import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

def bilateral_filter(image, d=9, sigma_color=75, sigma_space=75, plot_flag=False):
    """ Bilateral filter is a non-linear, edge-preserving, and noise-reducing smoothing filter.
    Args:
        - image: opencv image
        - d_values: Diameter of each pixel neighborhood.
        - sigma_colors: Filter sigma in color space.
        - sigma_spaces: Filter sigma in coordinate space.
        - pair_mode: if True, use corresponding indices (d[i], σ_color[i], σ_space[i])
                     instead of iterating over all combinations.
        - plot_flag: if True, plots the original and filtered images.
    Returns:
        - images: dictionary with (d, σ_color, σ_space) tuples as keys and filtered images as values.
    """
    filtered_image = cv2.bilateralFilter(image, d, sigma_color, sigma_space)

    if plot_flag:
        cmap = 'gray' if len(image.shape) == 2 else None
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].imshow(image, cmap=cmap)
        axs[0].set_title("Original Image")
        axs[0].axis('off')

        axs[1].imshow(filtered_image, cmap=cmap)
        axs[1].set_title(f'Bilateral Filter\n(d={d}, σc={sigma_color}, σs={sigma_space})')
        axs[1].axis('off')

        plt.tight_layout()
        plt.show()

    return filtered_image


def study_bilateral_filter(
    image,
    d_values=[5, 9, 15],
    sigma_colors=[75, 150, 250],
    sigma_spaces=[75, 150, 250],
    pair_mode=True,
    plot_flag=False
):
    """
    Study the bilateral effect for different entrance value

    Args:
        image (numpy.ndarray): opencv image.
        d_values (list[int]): Diámetros del vecindario.
        sigma_colors (list[float]): Array of sigma space colors. 
        sigma_spaces (list[float]): Array of sigma spacial colors.

    Returns:
        dict: set of filtered images, {(d,sigma_color, sigma_space):filtered_image}
    """
    images = {}

    if pair_mode:
        if not (len(d_values) == len(sigma_colors) == len(sigma_spaces)):
            raise ValueError("All parameter lists must have the same length when pair_mode=True.")
        for d, sigma_color, sigma_space in zip(d_values, sigma_colors, sigma_spaces):
            images[(d, sigma_color, sigma_space)] = bilateral_filter(
                image, d=d, sigma_color=sigma_color, sigma_space=sigma_space, plot_flag=False
            )
    else:
        for d in d_values:
            for sigma_color in sigma_colors:
                for sigma_space in sigma_spaces:
                    images[(d, sigma_color, sigma_space)] = bilateral_filter(
                        image, d=d, sigma_color=sigma_color, sigma_space=sigma_space, plot_flag=False
                    )

    if plot_flag:
        n = len(images) + 1
        rows = math.ceil(math.sqrt(n))
        cols = math.ceil(n / rows)
        fig, axs = plt.subplots(rows, cols, figsize=(15, 10))
        axs = axs.flatten()
        cmap = 'gray' if len(image.shape) == 2 else None

        axs[0].imshow(image, cmap=cmap)
        axs[0].set_title("Original Image")
        axs[0].axis('off')

        for i, ((d, σc, σs), filtered) in enumerate(images.items(), start=1):
            axs[i].imshow(filtered, cmap=cmap)
            axs[i].set_title(f'Bilateral Filter\n(d={d}, σc={σc}, σs={σs})')
            axs[i].axis('off')

        for j in range(i + 1, len(axs)):
            axs[j].axis('off')

        plt.tight_layout()
        plt.show()

    return images
