import os, cv2

def read_image_from_cv2(img_path, mode=cv2.COLOR_BGR2RGB):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found at path: {img_path}")
    image = cv2.imread(img_path, mode)
    return image

def read_set_of_images(img_paths, mode=cv2.COLOR_BGR2RGB):
    images = []
    for img_path in img_paths:
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image not found at path: {img_path}")
        image = cv2.imread(img_path, mode)
        images.append(image)
    return images

