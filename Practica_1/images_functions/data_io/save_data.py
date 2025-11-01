import os, cv2


def save_image_cv2(image, save_path):
    if image is None:
        raise ValueError("Invalid image provided.")
    directory = os.path.dirname(save_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)
    cv2.imwrite(save_path, image)
