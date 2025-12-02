import os 
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import config
import matplotlib.pyplot as plt
import numpy as np


np.random.seed(config.RANDOM_SEED)
FOLDERS = os.listdir(config.IMG_PATH)


def get_imgs_paths(folder):
    """
    With this function we get the exast path of each image in the data folder
    """
    paths = []
    folder = os.path.join(config.IMG_PATH, folder)
    imgs = os.listdir(folder)
    for img in imgs:
        paths.append(os.path.join(folder, img))
    return paths

def dataLoader(imread_color = cv2.IMREAD_COLOR_RGB, flag_summary=False):
    """
    This function is used to load the images and labels of the dataset.

    Args:
        flag_summary (bool): if True, prints a summary of image sizes and
                             number of images per class.

    Returns:
        data (list): list of loaded images (as numpy arrays).
        labels (list): list of labels corresponding to each image.
    """
    data = []
    labels = []

    if flag_summary:
        imgs_sizes = []
        num_imgs_per_class = {}  # dict: class_name -> num_images

    for label in FOLDERS:
        paths = get_imgs_paths(label)
        for path in paths:
            img = cv2.imread(path, imread_color)

            # Safety check in case an image cannot be read
            if img is None:
                print(f"Warning: could not read image {path}")
                continue

            data.append(img)
            labels.append(label)

            if flag_summary:
                imgs_sizes.append(img.shape)

        if flag_summary:
            num_imgs_per_class[label] = len(paths)

    if flag_summary:
        # Check if all images have the same size
        same_size = 1 if len(set(imgs_sizes)) == 1 else 0
        print(f"Has all the images same size? {'Yes' if same_size else 'No'}")

        if same_size:
            print(f"The size of all the images is: {imgs_sizes[0]}")
        else:
            unique_imgs_sizes = sorted(set(imgs_sizes))
            print("Different image sizes found:")
            for size in unique_imgs_sizes:
                print(f"  Size: {size}")

        print("\nNumber of images per class:")
        for label, count in num_imgs_per_class.items():
            print(f"  {label}: {count} images")

    return data, labels



def split_data(data, labels, train_ratio=0.7, val_ratio=0.1,
               balanced_labels=True, shuffle=True, flag_summary=False):
    """
    Divide los datos en train, validation y test.

    - Si train_ratio + val_ratio < 1.0  -> hay test (lo que sobra).
    - Si train_ratio + val_ratio == 1.0 -> NO hay test (test vacío).

    Args:
        data (array-like): lista o array de imágenes (N muestras).
        labels (array-like): lista o array de etiquetas (N muestras).
        train_ratio (float): proporción de datos para entrenamiento.
        val_ratio (float): proporción de datos para validación.
        balanced_labels (bool): si True, hace split estratificado por clase.
        shuffle (bool): si True, baraja los índices.
        flag_summary (bool): si True, imprime resumen de tamaños por split.

    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test
    """
    data = np.array(data)
    labels = np.array(labels)

    total_ratio = train_ratio + val_ratio
    assert total_ratio <= 1.0 + 1e-8, "train_ratio + val_ratio debe ser <= 1.0"

    no_test = np.isclose(total_ratio, 1.0)  # True cuando quieres solo train/val

    n_total = len(data)

    # --- Caso NO estratificado ---
    if not balanced_labels:
        indices = np.arange(n_total)
        if shuffle:
            np.random.shuffle(indices)

        if no_test:
            # Solo train y val, sin test
            n_train = int(n_total * train_ratio)
            n_val = n_total - n_train  # el resto va a validación

            train_idx = indices[:n_train]
            val_idx   = indices[n_train:n_train + n_val]
            test_idx  = np.array([], dtype=int)
        else:
            # Train / val / test (como antes)
            n_train = int(n_total * train_ratio)
            n_val   = int(n_total * val_ratio)

            train_idx = indices[:n_train]
            val_idx   = indices[n_train:n_train + n_val]
            test_idx  = indices[n_train + n_val:]

    else:
        # --- Caso estratificado por clase ---
        unique_labels = np.unique(labels)

        train_indices = []
        val_indices   = []
        test_indices  = []

        for lab in unique_labels:
            # Índices de esta clase
            cls_idx = np.where(labels == lab)[0]

            # Barajar índices de esta clase
            if shuffle:
                np.random.shuffle(cls_idx)

            n_cls = len(cls_idx)

            if no_test:
                # Split 2-way estratificado (train/val)
                n_train_cls = int(n_cls * train_ratio)
                n_val_cls   = n_cls - n_train_cls  # el resto va a validación

                cls_train = cls_idx[:n_train_cls]
                cls_val   = cls_idx[n_train_cls:n_train_cls + n_val_cls]
                cls_test  = np.array([], dtype=int)
            else:
                # Split 3-way estratificado (train/val/test)
                n_train_cls = int(n_cls * train_ratio)
                n_val_cls   = int(n_cls * val_ratio)

                cls_train = cls_idx[:n_train_cls]
                cls_val   = cls_idx[n_train_cls:n_train_cls + n_val_cls]
                cls_test  = cls_idx[n_train_cls + n_val_cls:]

            train_indices.extend(cls_train)
            val_indices.extend(cls_val)
            if not no_test:
                test_indices.extend(cls_test)

        train_idx = np.array(train_indices)
        val_idx   = np.array(val_indices)
        test_idx  = np.array(test_indices) if not no_test else np.array([], dtype=int)

        # Opcional: barajar dentro de cada split para mezclar clases
        if shuffle:
            np.random.shuffle(train_idx)
            np.random.shuffle(val_idx)
            if not no_test:
                np.random.shuffle(test_idx)

    # Construir splits finales
    X_train, y_train = data[train_idx], labels[train_idx]
    X_val,   y_val   = data[val_idx], labels[val_idx]
    X_test,  y_test  = data[test_idx], labels[test_idx]

    if flag_summary:
        print(f"Number of train images: {len(X_train)}")
        for label in np.unique(y_train):
            n_label = np.sum(y_train == label)
            print(f"  - {label}: {n_label} images")

        print(f"\nNumber of validation images: {len(X_val)}")
        for label in np.unique(y_val):
            n_label = np.sum(y_val == label)
            print(f"  - {label}: {n_label} images")

        print(f"\nNumber of test images: {len(X_test)}")
        if len(X_test) == 0:
            print("  (no test split)")
        else:
            for label in np.unique(y_test):
                n_label = np.sum(y_test == label)
                print(f"  - {label}: {n_label} images")

    return X_train, y_train, X_val, y_val, X_test, y_test


def load_split_data(imread_color = cv2.IMREAD_COLOR_RGB, 
                    train_ratio = 0.7, val_ratio = 0.1,
                    balanced_labels = True, shuffle = True,
                    flag_summary=False):
    """
    Load the dataset splited.
    """
    data, labels = dataLoader(imread_color, flag_summary)

    return split_data(data, labels, train_ratio, val_ratio,
                      balanced_labels, shuffle, flag_summary)

def plot_image(img, label=None, save_path=None):
    """
    Plot a single image with its label (optional).

    Args:
        img (np.array): imagen en formato BGR (como sale de cv2.imread).
        label (str, optional): etiqueta a mostrar como título.
        save_path (str, optional): si se pasa una ruta, guarda la figura en vez de mostrarla.
    """
    plt.figure(figsize=(5, 5), dpi=200)  # tamaño suficiente para una imagen 576x576

    # Convertir de BGR (OpenCV) a RGB (matplotlib)
    if img.ndim == 3 and img.shape[2] == 3:
        img_to_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_to_show)
    else:
        # Por si tienes imágenes en escala de grises
        plt.imshow(img, cmap='gray')

    plt.axis('off')

    if label is not None:
        plt.title(str(label))

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


    
def plot_N_images(data, labels, N=6, max_cols=4, balanced = False, random=True, save_path=None):
    """
    Visualize N images from the dataset in a subplot grid.

    Args:
        data (list or array): lista/array de imágenes.
        labels (list or array): lista/array de etiquetas.
        N (int): número de imágenes a visualizar.
        max_cols (int): número máximo de columnas para mantener las imágenes legibles.
        
        random (bool): si True, selecciona N imágenes aleatorias; si False, las N primeras.
        save_path (str, optional): si se pasa una ruta, guarda la figura en vez de mostrarla.
    """
    assert len(data) == len(labels), "data y labels deben tener la misma longitud"

    total = len(data)
    N = min(N, total)  # por si N > número de imágenes

    if random:
        indices = np.random.choice(total, size=N, replace=False)
    else:
        indices = np.arange(N)

    # Limitar el número de columnas para que las imágenes de 576x576 sigan siendo legibles
    # Con max_cols=3 y figsize=(cols*4, rows*4) se verán bien en la mayoría de pantallas.
    cols = min(max_cols, N)
    rows = int(np.ceil(N / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3), dpi=200)

    # Normalizar axes a una lista 1D para iterar cómodamente
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    else:
        axes = np.array(axes).reshape(-1)

    # Apagar todos los ejes por defecto
    for ax in axes:
        ax.axis('off')

    # Rellenar solo los primeros N
    for idx, ax in zip(indices, axes[:N]):
        img = data[idx]
        label = labels[idx]

        if img.ndim == 3 and img.shape[2] == 3:
            img_to_show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img_to_show)
        else:
            ax.imshow(img, cmap='gray')

        ax.set_title(str(label))
        ax.axis('off')

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    data, labels = dataLoader(cv2.IMREAD_GRAYSCALE,flag_summary=True)
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(data, labels, flag_summary=True)
    plot_N_images(X_train, y_train, N=16, max_cols=4, random=True)