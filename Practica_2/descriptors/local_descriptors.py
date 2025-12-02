import os
from typing import List, Tuple, Optional, Sequence, Dict, Any

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from tqdm import tqdm


# Optional global seed for reproducibility of random visualizations
np.random.seed(42)


# =============================================================================
# Local corner and edge descriptors
# =============================================================================


def harris_corner_features(
    image: np.ndarray,
    label: Any,
    block_size: int = 3,
    ksize: int = 5,
    k: float = 0.04,
    threshold_rel: float = 0.01,
    save_folder: Optional[str] = None,
    plot_flag: Optional[bool] = None,
) -> np.ndarray:
    """
    Compute Harris-corner based features for a single grayscale image.

    Descriptor:
        [ num_corners_total, # Numero total de esquinas
          corner_density_total, # Densidad de esquinas
          harris_max, # Maximo valor de la respuesta Harris
          harris_mean, # Media de la respuesta Harris
          harris_std, # Desviacion estandar de la respuesta Harris
          grid_corner_densities.flatten() ] # Densidad de esquinas por celda (grid_size)

    Args:
        image: 2D o 3D image. Si 3D, se convierte a escala de grises.
        label: etiqueta asociada a la imagen (solo para títulos).
        block_size: neighborhood size para Harris.
        ksize: aperture para Sobel.
        k: parámetro libre de Harris.
        threshold_rel: umbral relativo (0–1) sobre la respuesta máxima para
            decidir qué píxeles son esquinas.
        save_folder: carpeta donde guardar la visualización (si no es None).
        plot_flag: si True, muestra la figura; si False, la cierra.

    Returns:
        feature_vector: 1D np.ndarray con las características descritas.
    """
    # 1) Escala de grises
    if image.ndim == 3:
        image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        image_gray = image.copy()

    H, W = image_gray.shape
    gray_f32 = image_gray.astype(np.float32)

    # 2) Respuesta Harris
    harris_response = cv.cornerHarris(gray_f32, block_size, ksize, k)

    # Normalizada solo para visualizar
    harris_norm = cv.normalize(
        harris_response, None, 0, 255, cv.NORM_MINMAX, dtype=cv.CV_32FC1
    )

    # 3) Máscara de esquinas
    threshold = threshold_rel * harris_response.max()
    corner_mask = harris_response > threshold

    num_corners_total = int(np.sum(corner_mask))
    corner_density_total = num_corners_total / float(H * W)

    harris_max = float(harris_response.max())
    harris_mean = float(harris_response.mean())
    harris_std = float(harris_response.std())



    # 5) Feature vector
    feature_vector = np.hstack(
        [
            corner_density_total,
            harris_max,
            harris_mean,
            harris_std
        ]
    ).astype(np.float32)

    # 6) Visualización (igual que antes)
    if plot_flag or save_folder is not None:
        if image.ndim == 2:
            img_color = cv.cvtColor(image_gray, cv.COLOR_GRAY2BGR)
        else:
            img_color = image.copy()

        img_with_corners = img_color.copy()
        img_with_corners[corner_mask] = [0, 255, 0]  # verde

        fig, axs = plt.subplots(1, 3, figsize=(14, 7), dpi=200)

        axs[0].imshow(image_gray, cmap="gray")
        axs[0].set_title(f"Original - {label}")
        axs[0].axis("off")

        axs[1].imshow(harris_norm, cmap="gray")
        axs[1].set_title("Harris Response")
        axs[1].axis("off")

        img_rgb = cv.cvtColor(img_with_corners, cv.COLOR_BGR2RGB)
        axs[2].imshow(img_rgb)
        axs[2].set_title(
            "Harris Corners\n"
            f"#corners={num_corners_total}, dens={corner_density_total:.4f}"
        )
        axs[2].axis("off")

        plt.tight_layout()

        if save_folder is not None:
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, "harris_grid.png")
            plt.savefig(save_path, bbox_inches="tight")

        if plot_flag:
            plt.show()
        else:
            plt.close(fig)

    return feature_vector

def canny_edge_features(
    image: np.ndarray,
    label: Any,
    blur_ksize: Tuple[int, int] = (5, 5),
    blur_sigma: float = 1.5,
    thresholds: Tuple[int, int] = (50, 80),
    save_folder: Optional[str] = None,
    plot_flag: Optional[bool] = None,
) -> np.ndarray:
    """
    Compute Canny edge-based features for a single image.

    Descriptor (4D):
        [ edge_density_total,  # porcentaje de píxeles que son borde
          edge_max,            # valor máximo en la imagen de bordes
          edge_mean,           # media de la imagen de bordes
          edge_std ]           # desviación estándar de la imagen de bordes
    """

    # 1) Convertir a escala de grises
    if image.ndim == 3:
        gray_image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray_image = image.copy()

    H, W = gray_image.shape

    # 2) Gaussian blur
    blurred_image = cv.GaussianBlur(gray_image, blur_ksize, blur_sigma)

    # 3) Canny edges
    low, high = thresholds
    edges = cv.Canny(blurred_image, threshold1=low, threshold2=high)
    edge_mask = edges > 0

    num_edges_total = int(np.sum(edge_mask))
    area = float(H * W)
    edge_density_total = num_edges_total / area

    # Convertimos a float para estadísticos
    edges_f32 = edges.astype(np.float32)
    edge_max = float(edges_f32.max())
    edge_mean = float(edges_f32.mean())
    edge_std = float(edges_f32.std())

    # 4) Vector de características (4D)
    feature_vector = np.array(
        [
            edge_density_total,
            edge_max,
            edge_mean,
            edge_std,
        ],
        dtype=np.float32,
    )

    # 5) Visualización similar a Harris: original gris, blur, original + bordes en verde
    if plot_flag or save_folder is not None:
        # Imagen en color para pintar bordes encima
        if image.ndim == 2:
            img_color = cv.cvtColor(gray_image, cv.COLOR_GRAY2BGR)
        else:
            img_color = image.copy()

        img_with_edges = img_color.copy()
        img_with_edges[edge_mask] = [0, 255, 0]  # verde, igual que en Harris

        fig, axs = plt.subplots(1, 3, figsize=(14, 7), dpi=200)

        axs[0].imshow(gray_image, cmap="gray")
        axs[0].set_title(f"Original - {label}")
        axs[0].axis("off")

        axs[1].imshow(blurred_image, cmap="gray")
        axs[1].set_title("Gaussian Blur")
        axs[1].axis("off")

        img_rgb = cv.cvtColor(img_with_edges, cv.COLOR_BGR2RGB)
        axs[2].imshow(img_rgb)
        axs[2].set_title(
            "Canny Edges\n"
            f"(thr: {low}, {high})\n"
            f"#edges={num_edges_total}, dens={edge_density_total:.4f}"
        )
        axs[2].axis("off")

        plt.tight_layout()

        if save_folder is not None:
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, "canny_edges.png")
            plt.savefig(save_path, bbox_inches="tight")

        if plot_flag:
            plt.show()
        else:
            plt.close(fig)

    return feature_vector



# =============================================================================
# SIFT base function and 1D wrappers
# =============================================================================


def sift_features(
    image: np.ndarray,
    label: Optional[Any] = None,
    sift: Optional[cv.SIFT] = None,
    return_keypoints: bool = False,
    save_folder: Optional[str] = None,
    plot_flag: Optional[bool] = None,
):
    """
    Extract SIFT keypoints and descriptors from a single image.

    Args:
        image: 2D grayscale image.
        label: Optional label associated with the image (for titles).
        sift: Optional pre-created SIFT object. If None, a new one is created.
        return_keypoints: If True, both keypoints and descriptors are returned.
        save_folder: If not None, folder where the visualization will be saved.
        plot_flag: If True, `plt.show()` is called. If False, the figure is
            closed. If None, only saving (if `save_folder`) is performed.

    Returns:
        If return_keypoints is False:
            descriptors (np.ndarray of shape (N_kp, 128)) or None.
        If return_keypoints is True:
            keypoints (list of cv2.KeyPoint),
            descriptors (np.ndarray of shape (N_kp, 128)) or None.
    """
    if sift is None:
        sift = cv.SIFT_create()

    keypoints, descriptors = sift.detectAndCompute(image, None)

    if plot_flag or save_folder:
        image_with_keypoints = cv.drawKeypoints(
            image,
            keypoints,
            None,
            flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )

        plt.figure(figsize=(10, 10), dpi=200)
        plt.subplot(1, 2, 1)
        plt.imshow(image, cmap="gray")
        title = "Original Image"
        if label is not None:
            title += f" - Label {label}"
        plt.title(title)
        plt.axis("off")

        plt.subplot(1, 2, 2)
        plt.imshow(image_with_keypoints, cmap="gray")
        if descriptors is not None:
            desc_shape = descriptors.shape
            n_kp = len(keypoints)
        else:
            desc_shape = (0, 128)
            n_kp = 0
        plt.title(
            "Image with SIFT Keypoints: "
            f"{n_kp}\nDescriptors shape: {desc_shape}"
        )
        plt.axis("off")

        if save_folder:
            os.makedirs(save_folder, exist_ok=True)
            save_path = os.path.join(save_folder, "sift_keypoints.png")
            plt.savefig(save_path, bbox_inches="tight")

        if plot_flag:
            plt.show()
        else:
            plt.close()

    if return_keypoints:
        return keypoints, descriptors
    return descriptors


def sift_features_1D(
    image: np.ndarray,
    label: Optional[Any] = None,
    strategies: Sequence[str] = ("mean_std",),
    k: int = 50,
    sift: Optional[cv.SIFT] = None,
    save_folder: Optional[str] = None,
    plot_flag: Optional[bool] = None,
) -> np.ndarray:
    """
    Wrapper around `sift_features` to obtain a fixed-length 1D descriptor.

    Available strategies:
        - "mean":      Global mean of all SIFT descriptors (128 dims).
        - "mean_std":  Concatenate mean and std (256 dims).
        - "top_k":     Flatten the top-k keypoint descriptors by response
                       (k * 128 dims, padded with zeros if < k keypoints).

    Args:
        image: Grayscale image.
        label: Optional label (for plotting only).
        strategies: Iterable of strategies among {"mean", "mean_std", "top_k"}.
        k: Number of descriptors used when "top_k" is enabled.
        sift: Optional pre-created SIFT instance.
        save_folder: Folder where SIFT visualization will be stored.
        plot_flag: Whether to show the SIFT visualization.

    Returns:
        feature_vector_1d: Concatenated 1D numpy array.
    """
    keypoints, descriptors = sift_features(
        image,
        label=label,
        sift=sift,
        return_keypoints=True,
        save_folder=save_folder,
        plot_flag=plot_flag,
    )

    # Compute dimensionality for all requested strategies
    dim_per_strategy = []
    for strategy in strategies:
        if strategy == "mean":
            dim_per_strategy.append(128)
        elif strategy == "mean_std":
            dim_per_strategy.append(128 * 2)
        elif strategy == "top_k":
            dim_per_strategy.append(k * 128)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

    total_dim = sum(dim_per_strategy)

    # No descriptors -> return zero vector
    if descriptors is None or len(descriptors) == 0:
        return np.zeros(total_dim, dtype=np.float32)

    descriptors = descriptors.astype(np.float32)
    feature_parts = []

    for strategy in strategies:
        if strategy == "mean":
            part = np.mean(descriptors, axis=0).astype(np.float32)

        elif strategy == "mean_std":
            mu = np.mean(descriptors, axis=0)
            sigma = np.std(descriptors, axis=0)
            part = np.concatenate([mu, sigma], axis=0).astype(np.float32)

        elif strategy == "top_k":
            kp_desc = sorted(
                zip(keypoints, descriptors),
                key=lambda x: x[0].response,
                reverse=True,
            )
            desc_sorted = np.stack([d for (_, d) in kp_desc], axis=0)

            if desc_sorted.shape[0] >= k:
                desc_topk = desc_sorted[:k]
            else:
                pad = np.zeros(
                    (k - desc_sorted.shape[0], desc_sorted.shape[1]),
                    dtype=desc_sorted.dtype,
                )
                desc_topk = np.vstack([desc_sorted, pad])

            part = desc_topk.flatten().astype(np.float32)

        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        feature_parts.append(part)

    feature_vector_1d = np.concatenate(feature_parts, axis=0)
    return feature_vector_1d


def sift_features_set_images(
    images: Sequence[np.ndarray],
    sift: Optional[cv.SIFT] = None,
) -> List[np.ndarray]:
    """
    Extract SIFT descriptors from a collection of images and stack them
    into a single list (visual dictionary).

    Args:
        images: Iterable of 2D grayscale images.
        sift: Optional pre-created SIFT instance.

    Returns:
        descriptors_list: Python list of SIFT descriptors (each 128D),
            ready to be stacked with np.vstack for KMeans.
    """
    if sift is None:
        sift = cv.SIFT_create()

    dico: List[np.ndarray] = []
    for image in images:
        des = sift_features(image, sift=sift, return_keypoints=False)
        if des is not None and len(des) > 0:
            for v in des:
                dico.append(v.astype(np.float32))
    return dico


# =============================================================================
# SIFT codebook (KMeans)
# =============================================================================


def train_sift_codebook(
    dico: Sequence[np.ndarray],
    labels: Optional[Sequence[Any]] = None,
    n_clusters: Optional[int] = None,
    factor: int = 10,
    random_state: int = 0,
    n_init: Any = "auto",
) -> Tuple[KMeans, int, Optional[np.ndarray]]:
    """
    Train a SIFT codebook (KMeans) that can be shared by BoVW and VLAD.

    You can either:
        - specify `n_clusters` directly, or
        - provide `labels` and a `factor` so that:
              n_clusters = n_unique_labels * factor

    Args:
        dico: Iterable of SIFT descriptors (each 128D).
        labels: Optional label array to infer the number of clusters.
        n_clusters: Explicit number of clusters to use for KMeans.
        factor: If `n_clusters` is None and `labels` is not, then
            n_clusters = factor * n_unique_labels.
        random_state: Random seed passed to KMeans.
        n_init: n_init parameter for KMeans (e.g. "auto" or int).

    Returns:
        kmeans: Fitted KMeans model.
        k: Number of clusters.
        labels_unique: Array of unique labels or None.
    """
    dico_array = np.vstack(dico).astype(np.float32)

    labels_unique = None
    if n_clusters is None:
        if labels is None:
            raise ValueError(
                "Either `n_clusters` or `labels` must be provided to "
                "determine the number of clusters."
            )
        labels_unique = np.unique(labels)
        k = labels_unique.shape[0] * factor
    else:
        k = int(n_clusters)

    kmeans = KMeans(
        n_clusters=k,
        random_state=random_state,
        n_init=n_init,
    ).fit(dico_array)

    return kmeans, k, labels_unique




# =============================================================================
# Bag of Visual Words (BoVW)
# =============================================================================


def bovw_histogram_from_descriptors(
    des: Optional[np.ndarray],
    kmeans: KMeans,
    k: int,
) -> np.ndarray:
    """
    Compute a normalized BoVW histogram from a set of descriptors.

    Args:
        des: SIFT descriptors of one image (N_kp x 128) or None.
        kmeans: Trained KMeans codebook.
        k: Number of clusters (visual words).

    Returns:
        histo: 1D histogram of length k (L1-normalized).
    """
    histo = np.zeros(k, dtype=np.float32)
    if des is None or len(des) == 0:
        return histo

    des = des.astype(np.float32)
    idxs = kmeans.predict(des)
    counts = np.bincount(idxs, minlength=k).astype(np.float32)
    nkp = des.shape[0]
    if nkp > 0:
        histo = counts / float(nkp)
    return histo


def bovw_features_from_image(
    image: np.ndarray,
    kmeans: KMeans,
    k: int,
    sift: Optional[cv.SIFT] = None,
) -> np.ndarray:
    """
    Compute a BoVW histogram for a single image using SIFT.

    Args:
        image: Grayscale image.
        kmeans: Trained KMeans codebook.
        k: Number of clusters.
        sift: Optional SIFT instance.

    Returns:
        histo: 1D BoVW histogram of length k.
    """
    if sift is None:
        sift = cv.SIFT_create()

    des = sift_features(image, sift=sift, return_keypoints=False)
    histo = bovw_histogram_from_descriptors(des, kmeans, k)
    return histo


def plot_bovw_examples(
    images: Sequence[np.ndarray],
    labels: Sequence[Any],
    kmeans: KMeans,
    k: int,
    sift: Optional[cv.SIFT],
    indices: Optional[Sequence[int]] = None,
    N: int = 4,
    max_cols: int = 4,
    save_folder: Optional[str] = None,
    plot_flag: bool = False,
    label_map: Optional[Dict[Any, int]] = None,
):
    """
    Visualize example images together with their BoVW histograms.

    For each selected image, this function plots:
        - the image and its label
        - the BoVW histogram over the k visual words
    """
    if sift is None:
        sift = cv.SIFT_create()

    n_images = len(images)

    if indices is not None and len(indices) > 0:
        if min(indices) < 0 or max(indices) >= n_images:
            raise ValueError("Indices must be in the range [0, n_images-1].")
        idxs = list(indices)
    else:
        N = min(N, n_images)
        idxs = np.random.choice(n_images, N, replace=False)

    N = len(idxs)
    n_cols = min(N, max_cols)
    n_rows = int(np.ceil(N / n_cols))

    fig, axs = plt.subplots(2 * n_rows, n_cols, figsize=(4 * n_cols, 3 * 2 * n_rows), dpi=200)
    fig.suptitle("BoVW Examples")

    # Normalize axs to always be a 2D array: (2*n_rows, n_cols)
    if not isinstance(axs, np.ndarray):
        axs = np.array([[axs]])  # scalar Axes
    elif axs.ndim == 1:
        # case: n_cols == 1 -> shape (2*n_rows,) -> reshape to (2*n_rows, 1)
        axs = axs.reshape(-1, 1)

    for plot_idx, img_idx in enumerate(idxs):
        img = np.array(images[img_idx])
        lab = labels[img_idx]

        row_idx = plot_idx // n_cols
        col_idx = plot_idx % n_cols

        # row 0,2,4,... are images; row 1,3,5,... are histograms
        ax_img = axs[2 * row_idx, col_idx]
        ax_img.imshow(img, cmap="gray")
        ax_img.axis("off")

        if label_map is not None:
            ax_img.set_title(
                f"{lab}"
            )
        else:
            ax_img.set_title(f"{lab}")

        des = sift_features(img, sift=sift, return_keypoints=False)
        histo = bovw_histogram_from_descriptors(des, kmeans, k)

        ax_hist = axs[2 * row_idx + 1, col_idx]
        ax_hist.bar(range(k), histo)
        ax_hist.set_title("BoVW Histogram")
        ax_hist.set_xlabel("Cluster")
        ax_hist.set_ylabel("Frequency")
        ax_hist.set_xlim(0, k - 1)
        ax_hist.set_ylim(0, max(float(np.max(histo) * 1.1), 0.01))

    plt.tight_layout(rect=(0, 0, 1, 0.95))

    if save_folder:
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, "bovw_examples.png")
        plt.savefig(save_path, bbox_inches="tight")
    if plot_flag:
        plt.show()
    else:
        plt.close(fig)

    return fig


def build_bovw_dataset(
    images: Sequence[np.ndarray],
    labels: Sequence[Any],
    kmeans: KMeans,
    k: int,
    sift: Optional[cv.SIFT] = None,
    label_map: Optional[Dict[Any, int]] = None,
    save_folder: Optional[str] = None,
    plot_N: int = 4,
    plot_flag: bool = False,
    max_cols: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a BoVW dataset from a collection of images.

    Args:
        images: Sequence of grayscale images.
        labels: Corresponding labels.
        kmeans: Trained KMeans codebook.
        k: Number of clusters.
        sift: Optional SIFT instance.
        label_map: Optional dict mapping original labels to integer ids.
        save_folder: Folder where a visualization of BoVW examples is saved.
        plot_N: Number of random examples to visualize.
        plot_flag: Whether to show the visualization.
        max_cols: Max number of columns for visualization grid.

    Returns:
        X_bovw: Array of shape (N_images, k).
        y_bovw: Array of shape (N_images,) with integer-encoded labels.
    """
    if sift is None:
        sift = cv.SIFT_create()

    if label_map is None:
        unique = np.unique(labels)
        label_map = {label: i for i, label in enumerate(unique)}

    X_bovw = []
    y_bovw = []

    for img, lab in tqdm(zip(images, labels), total=len(images), desc="Building BoVW dataset"):
        des = sift_features(img, sift=sift, return_keypoints=False)
        histo = bovw_histogram_from_descriptors(des, kmeans, k)
        X_bovw.append(histo)
        y_bovw.append(label_map[lab])

    X_bovw = np.vstack(X_bovw).astype(np.float32)
    y_bovw = np.array(y_bovw).astype(np.int32)

    if plot_flag or save_folder or plot_N > 0:
        plot_bovw_examples(
            images=images,
            labels=labels,
            kmeans=kmeans,
            k=k,
            sift=sift,
            indices=None,
            N=plot_N,
            max_cols=max_cols,
            save_folder=save_folder,
            plot_flag=plot_flag,
            label_map=label_map,
        )

    return X_bovw, y_bovw


# =============================================================================
# VLAD
# =============================================================================


def vlad_from_descriptors(
    des: Optional[np.ndarray],
    kmeans: KMeans,
    normalize: bool = True,
    power_norm: bool = True,
    intra_norm: bool = False,
) -> np.ndarray:
    """
    Compute the VLAD vector from a set of local descriptors of ONE image.

    VLAD steps:
        1) Assign each descriptor to its nearest cluster center.
        2) For each center, accumulate residuals (x_i - mu_c).
        3) Concatenate all residuals into a single vector.
        4) Optionally apply power-normalization and L2-normalization.

    Args:
        des: SIFT descriptors of a single image (N_kp x D).
        kmeans: Trained KMeans codebook.
        normalize: Whether to apply final L2 normalization.
        power_norm: Whether to apply signed-sqrt (power) normalization.
        intra_norm: If True, L2-normalize per-cluster residuals before
            concatenation (intra-normalization).

    Returns:
        vlad: 1D VLAD vector of size (k * D,).
    """
    k = kmeans.n_clusters
    D = kmeans.cluster_centers_.shape[1]

    if des is None or len(des) == 0:
        return np.zeros(k * D, dtype=np.float32)

    des = des.astype(np.float32)
    assignments = kmeans.predict(des)

    residuals = np.zeros((k, D), dtype=np.float32)
    for c in range(k):
        mask = assignments == c
        if not np.any(mask):
            continue
        des_c = des[mask]
        diff = des_c - kmeans.cluster_centers_[c]
        residuals[c] = diff.sum(axis=0)

    if intra_norm:
        norms = np.linalg.norm(residuals, axis=1, keepdims=True) + 1e-12
        residuals = residuals / norms

    vlad = residuals.flatten()

    if power_norm:
        vlad = np.sign(vlad) * np.sqrt(np.abs(vlad))

    if normalize:
        norm = np.linalg.norm(vlad) + 1e-12
        vlad = vlad / norm

    return vlad.astype(np.float32)


def vlad_features_from_image(
    image: np.ndarray,
    kmeans: KMeans,
    sift: Optional[cv.SIFT] = None,
    normalize: bool = True,
    power_norm: bool = True,
    intra_norm: bool = False,
) -> np.ndarray:
    """
    Compute the VLAD descriptor for a single image using SIFT.

    Args:
        image: Grayscale image.
        kmeans: Trained KMeans codebook.
        sift: Optional SIFT instance.
        normalize: Whether to apply final L2 normalization.
        power_norm: Whether to apply signed-sqrt (power) normalization.
        intra_norm: Whether to use intra-normalization.

    Returns:
        vlad: 1D VLAD vector.
    """
    if sift is None:
        sift = cv.SIFT_create()

    des = sift_features(image, sift=sift, return_keypoints=False)
    vlad_vec = vlad_from_descriptors(
        des,
        kmeans,
        normalize=normalize,
        power_norm=power_norm,
        intra_norm=intra_norm,
    )
    return vlad_vec


def plot_vlad_examples(
    images: Sequence[np.ndarray],
    labels: Sequence[Any],
    kmeans: KMeans,
    sift: Optional[cv.SIFT] = None,
    indices: Optional[Sequence[int]] = None,
    N: int = 4,
    max_cols: int = 4,
    save_folder: Optional[str] = None,
    plot_flag: bool = False,
    label_map: Optional[Dict[Any, int]] = None,
    normalize: bool = True,
    power_norm: bool = True,
    intra_norm: bool = False,
):
    """
    Visualize example images together with their VLAD codeword activations.

    For each selected image, we compute its VLAD vector, reshape it as
    (k, D), and display the L2-norm of each cluster's residual vector as
    a "codeword activation" histogram.
    """
    if sift is None:
        sift = cv.SIFT_create()

    n_images = len(images)

    if indices is not None and len(indices) > 0:
        if min(indices) < 0 or max(indices) >= n_images:
            raise ValueError("Indices must be in the range [0, n_images-1].")
        idxs = list(indices)
    else:
        N = min(N, n_images)
        idxs = np.random.choice(n_images, N, replace=False)

    N = len(idxs)
    n_cols = min(N, max_cols)
    n_rows = int(np.ceil(N / n_cols))

    fig, axs = plt.subplots(2 * n_rows, n_cols, figsize=(4 * n_cols, 3 * 2 * n_rows), dpi=200)
    fig.suptitle("VLAD Examples (codeword activations)")

    # Normalize axs to always be a 2D array: (2*n_rows, n_cols)
    if not isinstance(axs, np.ndarray):
        axs = np.array([[axs]])  # scalar Axes
    elif axs.ndim == 1:
        axs = axs.reshape(-1, 1)

    k = kmeans.n_clusters
    D = kmeans.cluster_centers_.shape[1]

    for plot_idx, img_idx in enumerate(idxs):
        img = np.array(images[img_idx])
        lab = labels[img_idx]

        row_idx = plot_idx // n_cols
        col_idx = plot_idx % n_cols

        ax_img = axs[2 * row_idx, col_idx]
        ax_img.imshow(img, cmap="gray")
        ax_img.axis("off")

        if label_map is not None:
            ax_img.set_title(
                f"idx={img_idx} - label={lab} (enc={label_map.get(lab, '?')})"
            )
        else:
            ax_img.set_title(f"idx={img_idx} - label={lab}")

        des = sift_features(img, sift=sift, return_keypoints=False)
        vlad_vec = vlad_from_descriptors(
            des,
            kmeans,
            normalize=normalize,
            power_norm=power_norm,
            intra_norm=intra_norm,
        )

        vlad_2d = vlad_vec.reshape(k, D)
        cluster_norms = np.linalg.norm(vlad_2d, axis=1)

        ax_hist = axs[2 * row_idx + 1, col_idx]
        ax_hist.bar(range(k), cluster_norms)
        ax_hist.set_title("VLAD codeword norms")
        ax_hist.set_xlabel("Cluster")
        ax_hist.set_ylabel("L2 norm")

    plt.tight_layout(rect=(0, 0, 1, 0.95))

    if save_folder:
        os.makedirs(save_folder, exist_ok=True)
        save_path = os.path.join(save_folder, "vlad_examples.png")
        plt.savefig(save_path, bbox_inches="tight")
    if plot_flag:
        plt.show()
    else:
        plt.close(fig)

    return fig


def build_vlad_dataset(
    images: Sequence[np.ndarray],
    labels: Sequence[Any],
    kmeans: KMeans,
    sift: Optional[cv.SIFT] = None,
    label_map: Optional[Dict[Any, int]] = None,
    normalize: bool = True,
    power_norm: bool = True,
    intra_norm: bool = False,
    save_folder: Optional[str] = None,
    plot_N: int = 4,
    plot_flag: bool = False,
    max_cols: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a VLAD dataset from a collection of images using SIFT.

    Args:
        images: Sequence of grayscale images.
        labels: Corresponding labels.
        kmeans: Trained KMeans codebook.
        sift: Optional SIFT instance.
        label_map: Optional dict mapping original labels to integer ids.
        normalize, power_norm, intra_norm: Parameters passed to VLAD.
        save_folder: Folder where a visualization of VLAD examples is saved.
        plot_N: Number of random examples to visualize.
        plot_flag: Whether to show the visualization.
        max_cols: Max number of columns for visualization grid.

    Returns:
        X_vlad: Array of shape (N_images, k * D).
        y_vlad: Array of shape (N_images,) with integer-encoded labels.
    """
    if sift is None:
        sift = cv.SIFT_create()

    if label_map is None:
        unique = np.unique(labels)
        label_map = {label: i for i, label in enumerate(unique)}

    X_vlad = []
    y_vlad = []

    for img, lab in zip(images, labels):
        des = sift_features(img, sift=sift, return_keypoints=False)
        vlad_vec = vlad_from_descriptors(
            des,
            kmeans,
            normalize=normalize,
            power_norm=power_norm,
            intra_norm=intra_norm,
        )
        X_vlad.append(vlad_vec)
        y_vlad.append(label_map[lab])

    X_vlad = np.vstack(X_vlad).astype(np.float32)
    y_vlad = np.array(y_vlad).astype(np.int32)

    # Optional visualization, analogous to build_bovw_dataset
    if plot_flag or save_folder or plot_N > 0:
        plot_vlad_examples(
            images=images,
            labels=labels,
            kmeans=kmeans,
            sift=sift,
            indices=None,
            N=plot_N,
            max_cols=max_cols,
            save_folder=save_folder,
            plot_flag=plot_flag,
            label_map=label_map,
            normalize=normalize,
            power_norm=power_norm,
            intra_norm=intra_norm,
        )

    return X_vlad, y_vlad

