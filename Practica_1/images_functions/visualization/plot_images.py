import matplotlib.pyplot as plt
import os, random, math
from images_functions.data_io.read_data import read_image_from_cv2
from images_functions.data_io.read_imgs_directory import list_images_in_directory



def show_image_cv2(image):
    if image is None:
        raise ValueError("Invalid image provided.")
    cmap = 'gray' if len(image.shape) == 2 else None
    plt.imshow(image, cmap=cmap)
    plt.axis('off')
    plt.show()


def show_set_of_images(images, titles=None):
    n = len(images)
    t = len(titles) if titles is not None else 0
    if t != 0 and n != t:
        raise ValueError("Number of titles must match number of images.")
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n/cols)
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.flatten() if n > 1 else [axes]
    for i, ax in enumerate(axes):
        if i < n:
            img = images[i]
            cmap = 'gray' if len(img.shape) == 2 else None
            ax.imshow(img, cmap=cmap)
            if titles and i < t:
                ax.set_title(titles[i])
            ax.axis('off')
        else:
            ax.axis('off')
    plt.tight_layout()
    plt.show()

def show_N_random_images(img_path, num_images=5, mode=None):
    list_imgs = list_images_in_directory(img_path)
    if len(list_imgs) == 0:
        raise ValueError(f"No images found in directory: {img_path}")
    selected_imgs = random.sample(list_imgs, min(num_images, len(list_imgs)))
    n = len(selected_imgs)
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n/cols)
    fig, axes = plt.subplots(rows, cols, figsize=(10, 4 * rows))
    axes = axes.flatten() if num_images > 1 else [axes]
    for ax, img_path in zip(axes, selected_imgs):
        img = read_image_from_cv2(img_path, mode)
        ax.imshow(img)
        # ax.set_title(os.path.basename(img_path))
        ax.axis('off')
    for ax in axes[len(selected_imgs):]:
        ax.axis('off')
    plt.tight_layout()
    plt.show()
