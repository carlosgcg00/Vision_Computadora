import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import hog, graycomatrix, graycoprops
from skimage import exposure
import os
import cv2

def histogram_of_oriented_gradients(images, labels, orientations=9, pixels_per_cell=(8, 8),
                                    cells_per_block=(2, 2), block_norm='L2', visualize=True,
                                    feature_vector=True, save_folder=None, plot_flag=None,
                                    N=2, cols=2):
    """
    Compute the histogram of oriented gradients.
    Arguments:
        - images: list of images
        - labels: list of labels
        - orientations: number of orientations, bins in which the angles from 0 to 180 are divided
        - pixels_per_cell: number of pixels per cell
        - cells_per_block: number of cells per block
        - block_norm: normalization of the blocks it can be 'L2', 'L1'.
        - visualize: if we want to have the image of hog
        - feature_vector: if we want to have the feature vector in 1D

    Return:
        - hog_feature_list: HOG de todas las imágenes
        - hog_label_list: labels correspondientes
    """

    assert len(images) == len(labels), "images y labels deben tener la misma longitud"

    # Para reproducibilidad
    np.random.seed(42)

    hog_feature_list = []
    hog_label_list = []

    hog_visualizations = []  # solo para las que queramos plotear

    for img in images:
        hog_feat, hog_img = hog(
            img,
            orientations=orientations,
            pixels_per_cell=pixels_per_cell,
            cells_per_block=cells_per_block,
            block_norm=block_norm,
            visualize=True,
            feature_vector=True
        )

        hog_feature_list.append(hog_feat)
        hog_visualizations.append(exposure.rescale_intensity(hog_img, in_range=(0, 10)))

    idxs = np.random.choice(len(images), size=N, replace=False)

    total_cols = 2 * cols
    rows = int(np.ceil(N / cols))

    fig = plt.figure(figsize=(total_cols * 3, rows * 3))
    plot_index = 1

    for i, idx in enumerate(idxs):
        img = images[idx]
        label = labels[idx]
        hog_img = hog_visualizations[idx]
        hog_feat  = hog_feature_list[idx]


        # Imagen original
        ax1 = plt.subplot(rows, total_cols, plot_index)
        ax1.imshow(img, cmap='gray')
        ax1.set_title(f"{i} {label}")
        ax1.axis("off")
        plot_index += 1

        # Imagen HOG
        ax2 = plt.subplot(rows, total_cols, plot_index)
        ax2.imshow(hog_img, cmap='gray')
        ax2.set_title(f"{i} {label} HOG \n Features: {hog_feat.shape}")
        ax2.axis("off")
        plot_index += 1

    plt.tight_layout()

    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, "hog_grid.png")
        plt.savefig(save_path, dpi=200)

    if plot_flag:
        plt.show()
    else:
        plt.close(fig)

    return hog_feature_list


def glcm_global(images, labels, distances=[1],
                angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                save_folder=None, plot_flag=None, N=2, cols=2):
    """
    Compute the Gray Level Co ocurrence matrix.
    Arguments:
        - images: list of images
        - labels: list of labels
        - distances: list of distances, distance to neighbours where it is going to be computed
        - angles: list of angles, the direction where it is going to be computed
        - save_folder: folder to save the plots
        - plot_flag: if True, show the plots
        - N: number of images to plot
        - cols: number of columns in the plot
    Returns:
        - glcm_feature_list: list of glcm features  
    """
    glcm_feature_list = []

    # Para reproducibilidad
    np.random.seed(42)

    for image in images:
        glcm = graycomatrix(image, distances=distances, angles=angles,
                            levels=256, symmetric=True, normed=True)

        contrast = graycoprops(glcm, 'contrast')
        dissimilarity = graycoprops(glcm, 'dissimilarity')
        homogeneity = graycoprops(glcm, 'homogeneity')
        energy = graycoprops(glcm, 'energy')
        correlation = graycoprops(glcm, 'correlation')
        ASM = graycoprops(glcm, 'ASM')

        feature_vector = np.hstack([
            contrast.flatten(),
            dissimilarity.flatten(),
            homogeneity.flatten(),
            energy.flatten(),
            correlation.flatten(),
            ASM.flatten(),
        ])
        glcm_feature_list.append(feature_vector)

    idxs = np.random.choice(len(images), size=N, replace=False)

    for plot_i, idx in enumerate(idxs):

        img = images[idx]
        label = labels[idx]

        num_dist = len(distances)
        num_ang = len(angles)

        # +1 columna para colocar la imagen original
        fig = plt.figure(figsize=((num_ang + 1) * 3, num_dist * 3))
        fig.suptitle(f"GLCM Visualization - Image {plot_i} ({label})")

        plot_index = 1

        for d_i, d in enumerate(distances):

            # 1. PLOT DE LA IMAGEN ORIGINAL AL INICIO DE CADA FILA
            ax0 = plt.subplot(num_dist, num_ang + 1, plot_index)
            ax0.imshow(img, cmap='gray')
            ax0.set_title("Original")
            ax0.axis("off")
            plot_index += 1

            # 2. PLOTS DE GLCM POR ÁNGULO
            for a_i, a in enumerate(angles):

                glcm = graycomatrix(
                    img,
                    distances=[d],
                    angles=[a],
                    levels=256,
                    symmetric=True,
                    normed=True
                )

                con = graycoprops(glcm, 'contrast')[0][0]
                hom = graycoprops(glcm, 'homogeneity')[0][0]

                ax = plt.subplot(num_dist, num_ang + 1, plot_index)
                ax.imshow(glcm[:, :, 0, 0], cmap='gray')
                ax.set_title(
                    f"d={d}, angle={np.degrees(a):.0f}°\n"
                    f"Con={con:.2f}, Hom={hom:.2f}"
                )
                ax.axis("off")

                plot_index += 1

        plt.tight_layout()

        if save_folder is not None:
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, f"glcm_image_{plot_i}.png")
            plt.savefig(save_path, dpi=200)

        if plot_flag:
            plt.show()
        else:
            plt.close(fig)

    return glcm_feature_list


def glcm_with_patches(images, labels, distances=[1],
                      angles=[0, np.pi/4, np.pi/2, 3*np.pi/4],
                      patch_coords=None, grid_size=None,
                      save_folder=None, plot_flag=None,
                      N=2, cols=2):
    """
    Compute GLCM features using patches or grid division of each image.

    Arguments:
        - images: list of grayscale images.
        - labels: list of labels for each image.
        - distances: list of pixel distances for GLCM computation.
        - angles: list of angles (radians) for GLCM computation.
        - patch_coords: list of (y1, y2, x1, x2) defining manual patches.
        - grid_size: tuple (N, M) to split each image into N×M equal patches.
        - save_folder: folder to save visualization plots.
        - plot_flag: if True, display plots; if False, close them.
        - N: number of images to visualize (randomly chosen).
        - cols: not usado en el layout actual, se mantiene por compatibilidad.

    Returns:
        - feature_vectors: list with one 1D feature vector per image.
    """

    np.random.seed(42)
    N = min(N, len(images))
    idxs = np.random.choice(len(images), size=N, replace=False)

    all_feature_vectors = []

    # --------------------------------------------------------
    # LOOP PRINCIPAL SOBRE N IMÁGENES SELECCIONADAS
    # --------------------------------------------------------
    for plot_i, idx in enumerate(idxs):

        img = images[idx]
        label = labels[idx]
        H, W = img.shape

        # ----------------------------------------------------
        # 1) Definición de patches
        # ----------------------------------------------------
        if patch_coords is not None:
            coords_list = patch_coords

        elif grid_size is not None:
            r, c = grid_size
            y_step = H // r
            x_step = W // c
            coords_list = []
            for i in range(r):
                for j in range(c):
                    y1 = i * y_step
                    y2 = (i + 1) * y_step if i < r - 1 else H
                    x1 = j * x_step
                    x2 = (j + 1) * x_step if j < c - 1 else W
                    coords_list.append((y1, y2, x1, x2))
        else:
            raise ValueError("Debes proporcionar patch_coords o grid_size.")

        num_patches = len(coords_list)
        num_dist = len(distances)
        num_ang = len(angles)

        # ----------------------------------------------------
        # 2) Cálculo de features: un vector 1D por imagen
        # ----------------------------------------------------
        image_feature_vector = []
        for d in distances:
            for (y1, y2, x1, x2) in coords_list:
                patch = img[y1:y2, x1:x2]

                glcm = graycomatrix(
                    patch,
                    distances=[d],
                    angles=angles,
                    levels=256,
                    symmetric=True,
                    normed=True
                )

                contrast = graycoprops(glcm, 'contrast')
                dissimilarity = graycoprops(glcm, 'dissimilarity')
                homogeneity = graycoprops(glcm, 'homogeneity')
                energy = graycoprops(glcm, 'energy')
                correlation = graycoprops(glcm, 'correlation')
                ASM = graycoprops(glcm, 'ASM')

                fv = np.hstack([
                    contrast.flatten(),
                    dissimilarity.flatten(),
                    homogeneity.flatten(),
                    energy.flatten(),
                    correlation.flatten(),
                    ASM.flatten()
                ])
                image_feature_vector.append(fv)

        # Concatenar todos los patches y distancias en un solo vector 1D
        image_feature_vector = np.hstack(image_feature_vector)
        all_feature_vectors.append(image_feature_vector)

        # ----------------------------------------------------
        # 3) VISUALIZACIÓN:
        #    cada fila = un patch para una distancia:
        #    [IMG ORIGINAL + RECTÁNGULO] | [PATCH] | [GLCM ángulos...]
        # ----------------------------------------------------
        rows = num_patches * num_dist
        cols_plot = 2 + num_ang   # original+rect | patch | glcm(angle0) | ...

        fig, axes = plt.subplots(rows, cols_plot,
                                 figsize=(cols_plot * 3, rows * 3))
        fig.suptitle(f"Image {plot_i} - {label}")

        # Normalizar axes a 2D
        if rows == 1:
            axes = np.expand_dims(axes, axis=0)
        if cols_plot == 1:
            axes = np.expand_dims(axes, axis=1)

        row_idx = 0
        for d in distances:
            for p_idx, (y1, y2, x1, x2) in enumerate(coords_list):
                patch = img[y1:y2, x1:x2]

                # GLCM del patch (todas las orientaciones) para esta distancia
                glcm = graycomatrix(
                    patch,
                    distances=[d],
                    angles=angles,
                    levels=256,
                    symmetric=True,
                    normed=True
                )

                # Columna 0: imagen original con el patch marcado en rojo
                img_with_rect = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)
                cv2.rectangle(img_with_rect, (x1, y1), (x2, y2), (255, 0, 0), 2)
                ax_orig = axes[row_idx, 0]
                ax_orig.imshow(img_with_rect)
                ax_orig.set_title(
                    f"Patch {p_idx}, d={d}\n"
                    f"(x1={x1}, y1={y1}, x2={x2}, y2={y2})"
                )
                ax_orig.axis("off")

                # Columna 1: patch recortado
                ax_patch = axes[row_idx, 1]
                ax_patch.imshow(patch, cmap='gray')
                ax_patch.set_title("Patch")
                ax_patch.axis("off")

                # Columnas siguientes: GLCM por ángulo
                for a_idx, a in enumerate(angles):
                    ax = axes[row_idx, 2 + a_idx]
                    ax.imshow(glcm[:, :, 0, a_idx], cmap='gray')
                    ax.set_title(f"{int(np.degrees(a))}°")
                    ax.axis("off")

                row_idx += 1

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if save_folder is not None:
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, f"glcm_patches_img{plot_i}.png")
            plt.savefig(save_path, dpi=200)

        if plot_flag:
            plt.show()
        else:
            plt.close(fig)

    return all_feature_vectors



def lbp_pixel(image, y, x, start_from='top_left', debug=False):
    """
    Calcula el LBP de un píxel (y, x) usando vecindad 3x3.
    Ignora bordes: y y x deben estar en [1, H-2] y [1, W-2].
    """

    # Ventana 3x3 centrada en (y, x)
    window = image[y-1:y+1+1, x-1:x+1+1]  # shape (3,3)
    center = image[y, x]

    if start_from == 'top_left':
        # Orden de vecinos (en sentido horario, empezando arriba-izquierda):
        # p0 p1 p2
        # p7  C p3
        # p6 p4 p5
        p0 = image[y-1, x-1]
        p1 = image[y-1, x]
        p2 = image[y-1, x+1]
        p3 = image[y,   x+1]
        p4 = image[y+1, x+1]
        p5 = image[y+1, x]
        p6 = image[y+1, x-1]
        p7 = image[y,   x-1]

    elif start_from == 'top_right':
        # Orden de vecinos (en sentido horario, empezando arriba-derecha):
        # p6 p7 p0
        # p5  C p1
        # p4 p3 p2
        p6 = image[y-1, x-1]
        p7 = image[y-1, x]
        p0 = image[y-1, x+1]
        p1 = image[y,   x+1]
        p2 = image[y+1, x+1]
        p3 = image[y+1, x]
        p4 = image[y+1, x-1]
        p5 = image[y,   x-1]

    else:
        raise ValueError("start_from debe ser 'top_left' o 'top_right'.")

    neighbors = np.array([p0, p1, p2, p3, p4, p5, p6, p7], dtype=np.float32)

    # Comparación vecinas >= centro → bits 0/1
    bits = (neighbors >= center).astype(np.int32)

    # Pesos binarios 2^(i)
    weights = 2 ** np.arange(8)

    # Código LBP (decimal)
    code = int(np.sum(bits * weights))

    if debug:
        print("Ventana 3x3 alrededor de (y={}, x={}):".format(y, x))
        print(window)
        print("Centro:", center)
        print("Vecinos [p0..p7]:", neighbors)
        print("Bits    [p0..p7]:", bits)
        print("Código LBP (decimal):", code)

    return code


# ----------------------------------------------------------------------
#  LBP IMAGEN COMPLETA (MAPA 2D)
# ----------------------------------------------------------------------
def lbp_image_map(image, start_from='top_left'):
    """
    Calcula el mapa LBP de una imagen 2D (vecindario 3x3).
    Devuelve una imagen LBP de la misma shape, con bordes a 0.
    """
    H, W = image.shape
    lbp = np.zeros((H, W), dtype=np.uint8)

    for y in range(1, H-1):
        for x in range(1, W-1):
            lbp[y, x] = lbp_pixel(image, y, x, start_from=start_from, debug=False)

    return lbp


# ----------------------------------------------------------------------
#  DESCRIPTOR LBP PARA UN CONJUNTO DE IMÁGENES
# ----------------------------------------------------------------------
def local_binary_patterns(images, labels,
                          start_from='top_left',
                          save_folder=None, plot_flag=None,
                          N=2, cols=2):
    """
    Compute Local Binary Pattern (LBP) descriptors for a set of images.

    Arguments:
        - images: list of grayscale images.
        - labels: list of labels for each image.
        - start_from: 'top_left' or 'top_right' for the LBP neighbor ordering.
        - save_folder: folder to save visualization plots (one grid for all N).
        - plot_flag: if True, display plots; if False, close them.
        - N: number of images to visualize (randomly chosen).
        - cols: number of images per row in the visualization grid.

    Returns:
        - lbp_feature_list: list with one 1D LBP histogram (256 bins) per image.
    """

    assert len(images) == len(labels), "images y labels deben tener la misma longitud"

    # Para reproducibilidad en la selección de imágenes a mostrar
    np.random.seed(42)

    lbp_feature_list = []
    lbp_maps = []

    # 1) Cálculo de LBP y del histograma (descriptor) para cada imagen
    for img in images:
        lbp_map = lbp_image_map(img, start_from=start_from)
        lbp_maps.append(lbp_map)

        # 256 posibles patrones (0..255)
        hist, _ = np.histogram(
            lbp_map.ravel(),
            bins=256,
            range=(0, 256),
            density=True  # normalizado
        )
        lbp_feature_list.append(hist)

    # 2) Visualización de N imágenes
    N = min(N, len(images))
    idxs = np.random.choice(len(images), size=N, replace=False)

    total_cols = 3 * cols   # cada imagen ocupa 3 columnas: Original | LBP | Histograma
    rows = int(np.ceil(N / cols))

    fig = plt.figure(figsize=(total_cols * 3, rows * 3))
    plot_index = 1

    for i, idx in enumerate(idxs):
        img = images[idx]
        label = labels[idx]
        lbp_map = lbp_maps[idx]
        hist = lbp_feature_list[idx]

        # --- Imagen original ---
        ax1 = plt.subplot(rows, total_cols, plot_index)
        ax1.imshow(img, cmap='gray')
        ax1.set_title(f"{i} {label} - Original")
        ax1.axis("off")
        plot_index += 1

        # --- Imagen LBP ---
        ax2 = plt.subplot(rows, total_cols, plot_index)
        ax2.imshow(lbp_map, cmap='gray')
        ax2.set_title(f"{i} {label} - LBP")
        ax2.axis("off")
        plot_index += 1

        # --- Histograma LBP ---
        ax3 = plt.subplot(rows, total_cols, plot_index)
        ax3.plot(hist)
        ax3.set_title(f"{i} {label} - Histograma LBP")
        ax3.set_xlabel("Código LBP")
        ax3.set_ylabel("Frecuencia norm.")
        ax3.set_xlim(0, 255)
        plot_index += 1

    plt.tight_layout()

    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, "lbp_grid.png")
        plt.savefig(save_path, dpi=200)

    if plot_flag:
        plt.show()
    else:
        plt.close(fig)

    return lbp_feature_list
