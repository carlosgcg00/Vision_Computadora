import os, cv2

def list_images_in_directory(directory_path, valid_extensions={'.jpg', '.jpeg', '.png'}):
    if not os.path.isdir(directory_path):
        raise NotADirectoryError(f"Directory not found: {directory_path}")
    image_files = [os.path.join(directory_path, f) for f in os.listdir(directory_path)
                   if os.path.splitext(f)[1].lower() in valid_extensions]
    print(f"The directory {directory_path}, contains {len(image_files)} images")
    verify_all_images_same_size(image_files)
    return image_files

def verify_all_images_same_size(image_paths, mode=cv2.IMREAD_UNCHANGED):
    if not image_paths:
        raise ValueError("The image list is empty.")
    first_img = cv2.imread(image_paths[0], mode)
    if first_img is None:
        raise ValueError(f"Error reading first image: {image_paths[0]}")
    ref_shape = first_img.shape[:2]
    for path in image_paths[1:]:
        img = cv2.imread(path, mode)
        if img is None:
            raise ValueError(f"Error reading image: {path}")
        if img.shape[:2] != ref_shape:
            print(f"Image {os.path.basename(path)} has different size {img.shape[:2]}, expected {ref_shape}")
            return False
    print(f"All images have the same spatial size: {ref_shape}")
    return True
