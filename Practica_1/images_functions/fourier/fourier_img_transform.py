import cv2
import numpy as np
import matplotlib.pyplot as plt


def fourier_transform_image(image, plot_flag=False):
    """
    Computes the Fourier Transform of an image and returns the magnitude spectrum.

    Parameters:
    - image: Input image (grayscale).
    - plot_flag: If True, plots the original image and its magnitude spectrum.

    Returns:
    - magnitude_spectrum: The magnitude spectrum of the Fourier Transform.
    """
    # Compute the 2D Fourier Transform
    f_transform = np.fft.fft2(image)
    f_shifted = np.fft.fftshift(f_transform)

    # Compute the magnitude spectrum
    magnitude_spectrum = 20 * np.log(np.abs(f_shifted) + 1)  # Adding 1 to avoid log(0)

    if plot_flag:
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap='gray')
        plt.title('Original Image')
        plt.axis('off')

        plt.subplot(1, 2, 2)
        plt.imshow(magnitude_spectrum, cmap='gray')
        plt.title('Magnitude Spectrum')
        plt.axis('off')

        plt.show()

    return magnitude_spectrum, f_shifted


def inverse_fourier_transform(f_shifted, plot_flag=False):
    """
    Computes the Inverse Fourier Transform from the shifted Fourier Transform.

    Parameters:
    - f_shifted: Shifted Fourier Transform of the image.
    - plot_flag: If True, plots the reconstructed image.

    Returns:
    - reconstructed_image: The image reconstructed from the Inverse Fourier Transform.
    """
    # Shift back (inverse shift)
    f_ishifted = np.fft.ifftshift(f_shifted)

    # Compute the Inverse 2D Fourier Transform
    reconstructed_image = np.fft.ifft2(f_ishifted)
    reconstructed_image = np.abs(reconstructed_image)

    if plot_flag:
        plt.figure(figsize=(6, 6))
        plt.imshow(reconstructed_image, cmap='gray')
        plt.title('Reconstructed Image')
        plt.axis('off')
        plt.show()

    return reconstructed_image