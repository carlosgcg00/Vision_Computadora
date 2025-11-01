import cv2
import numpy as np
import matplotlib.pyplot as plt


def morphology_operations(image, operation='dilate', kernel_size=(5, 5), iterations=1, plot_flag=False, image_title='Original Image'):
    """
    Apply a selected morphological operation to a binarized image.

    Args:
        - image: Input binarized image (grayscale).
        - operation: Morphological operation to apply. 
                     Options: 'dilate', 'erode', 'open', 'close', 'gradient'.
        - kernel_size: Size of the structuring element (default is (5,5)).
        - iterations: Number of times the operation is applied (default is 1).
        - plot_flag: If True, plot the original and processed image.
        - image_title: Title for the original image.
    
    Returns:
        - morphed_image: Resulting image after applying the selected morphological operation.
    """
    # Define kernel
    kernel = np.ones(kernel_size, np.uint8)

    # Select operation
    operation = operation.lower()
    if operation == 'dilate':
        morphed_image = cv2.dilate(image, kernel, iterations=iterations)
    elif operation == 'erode':
        morphed_image = cv2.erode(image, kernel, iterations=iterations)
    elif operation == 'open':
        morphed_image = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
    elif operation == 'close':
        morphed_image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    elif operation == 'gradient':
        dilated = cv2.dilate(image, kernel, iterations=iterations)
        eroded = cv2.erode(image, kernel, iterations=iterations)
        morphed_image = cv2.subtract(dilated, eroded)
    else:
        raise ValueError("Invalid operation. Choose from: 'dilate', 'erode', 'open', 'close', 'gradient'.")

    # Optional plotting
    if plot_flag:
        fig, axs = plt.subplots(1, 2, figsize=(8, 10))
        axs[0].set_title(image_title)
        axs[0].imshow(image, cmap='gray')
        axs[0].axis('off')

        axs[1].set_title(f'Morphological Operation: {operation.capitalize()}')
        axs[1].imshow(morphed_image, cmap='gray')
        axs[1].axis('off')

        plt.tight_layout()
        plt.show()

    return morphed_image



def study_morphology_operations(image, kernel_size=(5, 5), iterations=1, plot_flag=False, image_title='Original Image'):
    """
    Apply morphological operations to a binarized image.
    Args:
        - image: Input binarized image (grayscale).
        - kernel_size: Size of the structuring element (default is (5,5)).
        - iterations: Number of times the operation is applied (default is 1).
        - plot_flag: If True, plot the original and processed images.
    Returns:
        - morphed_images: Dictionary containing the results of each morphological operation.
    """
    # Define the kernel
    kernel = np.ones(kernel_size, np.uint8)
    
    dilated = cv2.dilate(image, kernel, iterations=iterations)
    eroded = cv2.erode(image, kernel, iterations=iterations)
    opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel, iterations=iterations)
    closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    morphologic_gradient = cv2.subtract(dilated, eroded)

    morphed_images = {'dilated': dilated, 'eroded': eroded, 'opened': opened, 'closed': closed, 'morphologic_gradient': morphologic_gradient}

    if plot_flag:
        # Plot using axs
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        axs = axs.ravel()

        titles = [image_title,'Dilated','Eroded','Opened','Closed','Morphologic Gradient']
        images = [image,dilated,eroded,opened,closed,morphologic_gradient]

        for ax, title, img in zip(axs, titles, images):
            ax.set_title(title)
            ax.imshow(img, cmap='gray')
            ax.axis('off')

        # Hide any empty subplots if odd count
        for ax in axs[len(images):]:
            ax.axis('off')

        plt.tight_layout()
        plt.show()

    return morphed_images
