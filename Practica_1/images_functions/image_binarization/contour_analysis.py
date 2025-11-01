import cv2
import numpy as np
import matplotlib.pyplot as plt

def find_contours_img_binarization(
    image,
    binarized_image,
    mode=cv2.RETR_EXTERNAL,
    method=cv2.CHAIN_APPROX_SIMPLE,
    area_relative_min=0.0,
    area_max=False,           # ← NUEVO PARÁMETRO: activa el filtrado restrictivo
    plot_flag=False,
    save_plot=None,
    image_title='Original Image',
    binarized_image_title='Binarized Image'
):
    """
    Finds contours in a binarized image, filters them by area, and optionally
    keeps only the largest contour if area_max=True.

    Args:
        image (np.ndarray): Original grayscale image.
        binarized_image (np.ndarray): Binarized (thresholded) image.
        mode (int): Contour retrieval mode (default cv2.RETR_EXTERNAL).
        method (int): Contour approximation method (default cv2.CHAIN_APPROX_SIMPLE).
        area_relative_min (float): Minimum relative area (fraction of total image area).
        area_max (bool): If True, ignores area_relative_min and keeps only the largest contour.
        plot_flag (bool): If True, display the results with matplotlib.
        save_plot (str or None): If provided, saves the visualization to this path.

    Returns:
        tuple:
            - contours (list): Filtered contours.
            - centroids (list): List of (x, y) centroids.
            - areas (list): Contour areas.
            - bounding_boxes (list): Bounding boxes as (x, y, w, h).
            - contour_masks (list): Boolean masks for each contour.
    """

    if image is None or binarized_image is None:
        raise ValueError("Input images cannot be None.")

    height, width = image.shape[:2]
    total_area = height * width

    # Detect contours
    contours, _ = cv2.findContours(binarized_image, mode, method)
    if not contours:
        return [], [], [], [], []

    # Filter by relative area (if not in "max only" mode)
    if not area_max:
        area_threshold = total_area * area_relative_min
        filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= area_threshold]
    else:
        # Mantener solo el contorno con el área máxima
        max_contour = max(contours, key=cv2.contourArea)
        filtered_contours = [max_contour]

    centroids, areas, bounding_boxes, contour_masks = [], [], [], []
    mask_vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    img_binarized_filtered = np.zeros((height, width), dtype=bool)

    for cnt in filtered_contours:
        M = cv2.moments(cnt)
        if M["m00"] == 0:
            continue

        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)

        centroids.append((cX, cY))
        areas.append(area)
        bounding_boxes.append((x, y, w, h))

        mask_i = np.zeros((height, width), dtype=np.uint8)
        cv2.drawContours(mask_i, [cnt], -1, 255, -1)
        contour_masks.append(mask_i.astype(bool))
        img_binarized_filtered = np.logical_or(img_binarized_filtered, mask_i)

        if plot_flag or save_plot:
            cv2.drawContours(mask_vis, [cnt], -1, (0, 255, 0), 2)
            cv2.circle(mask_vis, (cX, cY), 5, (255, 0, 0), -1)
            cv2.rectangle(mask_vis, (x, y), (x + w, y + h), (0, 0, 255), 2)

    # Visualization
    if plot_flag or save_plot:
        fig, axs = plt.subplots(1, 4, figsize=(15, 5))
        axs[0].set_title(image_title)
        axs[0].imshow(image, cmap='gray')
        axs[0].axis('off')

        axs[1].set_title(binarized_image_title)
        axs[1].imshow(binarized_image, cmap='gray')
        axs[1].axis('off')

        if area_max:
            title = 'Largest Contour Only'
        else:
            title = f'Contours (min area = {area_relative_min:.4f})'
        axs[2].set_title(title)
        axs[2].imshow(mask_vis)
        axs[2].axis('off')

        
        axs[3].set_title('Filtered Binarized Image')
        axs[3].imshow(img_binarized_filtered, cmap='gray')
        axs[3].axis('off')


        plt.tight_layout()

        if save_plot:
            plt.savefig(save_plot, bbox_inches='tight')

        if plot_flag:
            plt.show()
        else:
            plt.close(fig)

    return filtered_contours, centroids, areas, bounding_boxes, contour_masks, img_binarized_filtered
