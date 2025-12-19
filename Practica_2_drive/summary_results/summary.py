# summary.py
import os
from datetime import datetime

import numpy as np
import pandas as pd


def build_experiment_row(
    name_model: str,
    metrics_val: dict,
    num_vars: int = np.nan,
    trainable_params: int = np.nan,
    steps = None,
    process_name: str = ""
) -> pd.DataFrame:
    """
    Construye un DataFrame con UNA fila describiendo un experimento.

    Columnas:
      - timestamp
      - model_name
      - pipeline_features (titles de steps unidos con ' - ')
      - accuracy_val, precision_val, recall_val, f1_val
    """

    # Nombre de las features de la pipeline (usamos el title)
    feature_names = []
    if steps:
        for func, title, params in steps:
            feature_names.append(title if title is not None else func.__name__)

        pipeline_features = " - ".join(feature_names)
    else:
        pipeline_features = process_name


    row_dict = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model_name": name_model,
        "pipeline_features": pipeline_features,
        "num_vars": num_vars,
        "trainable_params": trainable_params,
        "accuracy_val": round(metrics_val.get("accuracy", np.nan), 2),
        "precision_val": round(metrics_val.get("precision", np.nan), 2),
        "recall_val": round(metrics_val.get("recall", np.nan), 2),
        "f1_val": round(metrics_val.get("f1", np.nan), 2),
    }

    df_row = pd.DataFrame([row_dict])
    return df_row


def save_experiment_summary(csv_file: str, df_row: pd.DataFrame) -> pd.DataFrame:
    """
    Añade la fila de experimento a un CSV (creándolo si no existe).

    Parámetros
    ----------
    csv_file : str
        Ruta al archivo CSV donde se acumulan los resultados.
    df_row : pd.DataFrame
        DataFrame con UNA fila (salida de build_experiment_row).

    Devuelve
    --------
    df_all : pd.DataFrame
        DataFrame completo con todos los experimentos tras la inserción.
    """

    if os.path.exists(csv_file):
        df_all = pd.read_csv(csv_file)
        df_all = pd.concat([df_all, df_row], ignore_index=True)
    else:
        df_all = df_row.copy()

    df_all.to_csv(csv_file, index=False)
    return df_all
