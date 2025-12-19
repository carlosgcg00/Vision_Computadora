import numpy as np
from tqdm import tqdm

def feature_extraction_image(img, label, steps, debug=True):
    """
    Sequentially applies a set of feature-extraction functions to an image,
    concatenating their 1D outputs into a single feature vector.

    Args:
        img (numpy.ndarray): Input image (típicamente en escala de grises).
        label (str or int): Label asociada a la imagen; se pasa a cada función.
        steps (list): Lista de tuplas (func, title, params_dict).
                      Cada 'func' debe tener firma tipo:
                          func(img, label, **params) -> np.ndarray 1D
        debug (bool): Si True, imprime por consola cuántos parámetros
                      añade cada paso y el total acumulado.

    Returns:
        feature_vector (numpy.ndarray): Vector 1D con todas las features concatenadas.
        feature_info (list): Lista de diccionarios con metadatos por capa:
                             - step
                             - func_title
                             - func
                             - params
                             - dim
                             - start_idx
                             - end_idx
                             - cumulative_dim
    """

    features = []
    feature_info = []

    cumulative_dim = 0

    # Estado inicial (sin features)
    feature_info.append({
        "step": 0,
        "func_title": "Start (no features)",
        "func": "input",
        "params": {},
        "dim": 0,
        "start_idx": 0,
        "end_idx": 0,
        "cumulative_dim": 0,
        "label": label
    })

    for i, (func, func_title, params) in enumerate(steps, start=1):
        params = (params or {}).copy()

        step_features = func(img, label, **params)


        dim = step_features.shape[0]
        start_idx = cumulative_dim
        end_idx = cumulative_dim + dim
        cumulative_dim = end_idx

        features.append(step_features)

        feature_info.append({
            "step": i,
            "func_title": func_title,
            "func": func.__name__,
            "params": params,
            "dim": dim,
            "start_idx": start_idx,
            "end_idx": end_idx,      
            "cumulative_dim": cumulative_dim
        })

    # Concatenación final
    if features:
        feature_vector = np.concatenate(features, axis=0)
    else:
        feature_vector = np.array([], dtype=float)

    # Debug por consola (opcional)
    if debug:
        print("=== Feature pipeline summary ===")
        for info in feature_info[1:]:  # saltamos el paso 0
            print(f"Step {info['step']} - {info['func_title']} ({info['func']}): dim={info['dim']}.")
        print(f"Total dimensionality: {feature_vector.shape[0]}")

    return feature_vector, feature_info

def pipeline_feature_extraction(images, labels, steps, debug=False):
    """
    It will apply the feature extraction pipeline to a list of images.
    
    Args:
        images (list): List of images to extract features from.
        labels (list): List of labels for each image.
        steps (list): List of steps to apply to the images.
        debug (bool): If True, it will print debug information.
    
    Returns:
        features (list): List of features for each image.
        feature_info (list): List of feature information for each image.
    """
    features = []
    feature_info = []
    
    for img, label in tqdm(zip(images, labels), total=len(images), desc="Extracting features"):
        feature_vector, feature_info = feature_extraction_image(img, label, steps, debug=debug)
        features.append(feature_vector)
    
    return features, feature_info