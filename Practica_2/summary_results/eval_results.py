# eval_results.py
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.decomposition import PCA


def compute_metrics(y_true, y_pred, average="macro", verbose=True):
    """
    Calcula métricas estándar de clasificación y las devuelve en un dict.
    Si verbose=True, las imprime por pantalla.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average=average)
    rec = recall_score(y_true, y_pred, average=average)
    f1 = f1_score(y_true, y_pred, average=average)

    if verbose:
        print(f"Acc: {acc:.2f}")
        print(f"Prec: {prec:.2f}")
        print(f"Rec: {rec:.2f}")
        print(f"F1: {f1:.2f}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }

def plot_confusion_matrix(
    matrix,
    classes=None,
    cmap="viridis",
    title=None,
    ax=None,
    plot_flag=True,
    save_folder=None,
    file_name=None,
    show=True,
):
    """
    Dibuja una matriz de confusión.

    Parámetros:
      - matrix: np.ndarray con la matriz de confusión.
      - classes: lista de nombres de clase.
      - cmap: colormap de matplotlib.
      - title: título opcional.
      - ax: axis de matplotlib; si es None, crea una figura nueva.
      - plot_flag: si False, no se dibuja nada (solo se devuelve).
      - save_folder: si no es None y se ha creado figura, guarda la imagen en esa ruta.
      - file_name: name of the file to save the plot
      - show: si True y se ha creado figura, hace plt.show(); si False, solo cierra.
    """
    if not plot_flag:
        # No dibujamos nada, simplemente salimos.
        return

    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
        created_fig = True
    else:
        fig = ax.figure

    # Heatmap
    im = ax.imshow(matrix, interpolation="nearest", cmap=cmap)
    if created_fig:
        fig.colorbar(im, ax=ax)
    else:
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Clases
    if classes is None:
        classes = np.arange(matrix.shape[0])

    ax.set_xticks(np.arange(len(classes)))
    ax.set_yticks(np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45)
    ax.set_yticklabels(classes)

    # Eje X abajo (ticks y etiqueta)
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Valor real")
    ax.xaxis.set_ticks_position("bottom")
    ax.xaxis.set_label_position("bottom")

    # Números dentro de las celdas
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                va="center",
                ha="center",
                color="white" if matrix[i, j] > matrix.max() / 2 else "black",
            )

    if title:
        ax.set_title(title)

    if created_fig:
        plt.tight_layout()

        if save_folder is not None:
            os.makedirs(save_folder, exist_ok=True)
            file_name = file_name if file_name is not None else "confusion_matrix.png"
            save_path = os.path.join(save_folder, file_name)
            fig.savefig(save_path, dpi=200, bbox_inches="tight")

        if show:
            plt.show()
        else:
            plt.close(fig)


def compute_pca_train_transform_val(
    X_train,
    X_val,
    pipeline=None,
    n_components=2,
):
    """
    Ajusta un PCA sobre X_train (opcionalmente tras aplicar el scaler de un pipeline)
    y transforma X_train y X_val con ese mismo PCA.

    Devuelve:
      - X_train_2d, X_val_2d, pca
    """
    # Preprocesado: si hay pipeline con scaler, lo usamos
    if pipeline is not None and "scaler" in pipeline.named_steps:
        scaler = pipeline.named_steps["scaler"]
        X_train_proc = scaler.transform(X_train)
        X_val_proc = scaler.transform(X_val)
    else:
        X_train_proc = X_train
        X_val_proc = X_val

    pca = PCA(n_components=n_components)
    pca.fit(X_train_proc)

    X_train_2d = pca.transform(X_train_proc)
    X_val_2d = pca.transform(X_val_proc)

    return X_train_2d, X_val_2d, pca


def plot_pca_scatter_val(
    X_val_2d,
    y_val,
    y_pred,
    classes=None,
    title=None,
    ax=None,
):
    """
    Dibuja el scatter de PCA (validación) donde:
      - O = muestras correctamente clasificadas
      - X = errores
    """
    created_fig = False
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=200)
        created_fig = True

    y_val = np.asarray(y_val)
    y_pred = np.asarray(y_pred)

    correct = (y_val == y_pred)

    if classes is None:
        classes = np.unique(y_val)

    for cls in classes:
        idx_cls = (y_val == cls)
        idx_ok = idx_cls & correct
        idx_err = idx_cls & ~correct

        # Correctos
        ax.scatter(
            X_val_2d[idx_ok, 0],
            X_val_2d[idx_ok, 1],
            marker="o",
            alpha=0.7,
            label=f"{cls} (ok)",
        )

        # Errores
        if np.any(idx_err):
            ax.scatter(
                X_val_2d[idx_err, 0],
                X_val_2d[idx_err, 1],
                marker="x",
                s=80,
                linewidths=1.5,
                alpha=0.9,
                label=f"{cls} (err)",
            )

    ax.set_xlabel("PCA 1 (train)")
    ax.set_ylabel("PCA 2 (train)")
    if title:
        ax.set_title(title)

    # Evitar duplicar legendas si este plot se reutiliza en subplots
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), bbox_to_anchor=(1.05, 1), loc="upper left")

    if created_fig:
        plt.tight_layout()
        plt.show()


def evaluate_and_plot(
    X_train,
    X_val,
    y_val,
    y_pred,
    pipeline=None,
    class_names=None,
    fig_title=None,
    save_folder=None,
    file_name=None,
    show=True,
):
    """
    - Calcula métricas y matriz de confusión en validación.
    - Ajusta PCA en train (con el mismo escalado que vea el modelo, si pipeline tiene scaler)
      y lo aplica a validación.
    - Dibuja en una figura:
        [0] matriz de confusión
        [1] scatter PCA con aciertos/errores
    - Devuelve (metrics_dict, confusion_matrix).
    """

    # 1) Métricas + matriz de confusión
    metrics = compute_metrics(y_val, y_pred, verbose=True)
    cm = confusion_matrix(y_val, y_pred)

    # 2) PCA (entrenado SOLO con train)
    X_train_arr = np.asarray(X_train)
    X_val_arr = np.asarray(X_val)

    X_train_2d, X_val_2d, _ = compute_pca_train_transform_val(
        X_train_arr,
        X_val_arr,
        pipeline=pipeline,
        n_components=2,
    )

    # 3) Figura con subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=200)

    plot_confusion_matrix(
        cm,
        classes=class_names,
        cmap="Blues",
        title="Matriz de confusión (val)",
        ax=axes[0],
    )

    plot_pca_scatter_val(
        X_val_2d,
        y_val,
        y_pred,
        classes=class_names,
        title="PCA (train) aplicado a validación",
        ax=axes[1],
    )

    if fig_title:
        fig.suptitle(fig_title)

    plt.tight_layout()

    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)
        file_name = file_name if file_name is not None else "confusion_matrix.png"
        save_path = os.path.join(save_folder, file_name)
        fig.savefig(save_path, dpi=200)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return metrics, cm


def evaluate_and_plot_pca3(
    X_train,
    X_val,
    y_val,
    y_pred,
    pipeline=None,
    class_names=None,
    fig_title=None,
    save_folder=None,
    file_name=None,
    show=True,
):
    """
    Igual que evaluate_and_plot, pero con PCA de 3 componentes y
    3 scatterplots: (PC1,PC2), (PC2,PC3), (PC1,PC3) en una figura 2x2.
    """

    # 1) Métricas + matriz de confusión
    metrics = compute_metrics(y_val, y_pred, verbose=True)
    cm = confusion_matrix(y_val, y_pred)

    # 2) PCA (3 componentes) entrenado solo con train
    X_train_arr = np.asarray(X_train)
    X_val_arr = np.asarray(X_val)

    X_train_3d, X_val_3d, _ = compute_pca_train_transform_val(
        X_train_arr,
        X_val_arr,
        pipeline=pipeline,
        n_components=3,
    )

    y_val = np.asarray(y_val)
    y_pred = np.asarray(y_pred)
    correct = (y_val == y_pred)
    if class_names is None:
        class_names = np.unique(y_val)

    # 3) Figura 2x2
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=200)

    # (0,0): matriz de confusión
    plot_confusion_matrix(
        cm,
        classes=class_names,
        cmap="Blues",
        title="Matriz de confusión (val)",
        ax=axes[0, 0],
    )

    # Función interna simple para dibujar un plano PCi–PCj
    def _scatter_pc(ax, i, j, title):
        for cls in class_names:
            idx_cls = (y_val == cls)
            idx_ok = idx_cls & correct
            idx_err = idx_cls & ~correct

            # Correctos
            ax.scatter(
                X_val_3d[idx_ok, i],
                X_val_3d[idx_ok, j],
                marker="o",
                alpha=0.7,
                label=f"{cls} (ok)",
            )
            # Errores
            if np.any(idx_err):
                ax.scatter(
                    X_val_3d[idx_err, i],
                    X_val_3d[idx_err, j],
                    marker="x",
                    s=80,
                    linewidths=1.5,
                    alpha=0.9,
                    label=f"{cls} (err)",
                )
        ax.set_xlabel(f"PC{i+1}")
        ax.set_ylabel(f"PC{j+1}")
        ax.set_title(title)

    # (0,1): PC1 vs PC2
    _scatter_pc(axes[0, 1], 0, 1, "PCA: PC1 vs PC2")
    # (1,0): PC2 vs PC3
    _scatter_pc(axes[1, 0], 1, 2, "PCA: PC2 vs PC3")
    # (1,1): PC1 vs PC3
    _scatter_pc(axes[1, 1], 0, 2, "PCA: PC1 vs PC3")

    # Leyenda única (evitamos duplicados)
    handles, labels = axes[0, 1].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(
        by_label.values(),
        by_label.keys(),
        loc="upper center",
        ncol=min(len(by_label), 6),
        bbox_to_anchor=(0.5, 0.98),
    )

    if fig_title:
        fig.suptitle(fig_title, y=0.99)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)
        file_name = file_name if file_name is not None else "confusion_matrix.png"
        save_path = os.path.join(save_folder, file_name)
        fig.savefig(save_path, dpi=200)

    if show:
        plt.show()
    else:
        plt.close(fig)

    return metrics, cm



def plot_acc_loss(history, save_folder=None, file_name=None, show=True):
    """
    Plots the accuracy and loss of a model during training.
    """
    fig, ax = plt.subplots(1, 2, figsize=(12, 5), dpi=200)
    ax[0].plot(history.history['accuracy'])
    ax[0].plot(history.history['val_accuracy'])
    ax[0].set_title('Model accuracy')
    ax[0].set_ylabel('Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].legend(['Train', 'Validation'], loc='upper left')
    ax[1].plot(history.history['loss'])
    ax[1].plot(history.history['val_loss'])
    ax[1].set_title('Model loss')
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].legend(['Train', 'Validation'], loc='upper left')
    plt.tight_layout()
    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)
        file_name = file_name if file_name is not None else "acc_loss.png"
        save_path = os.path.join(save_folder, file_name)
        fig.savefig(save_path, dpi=200)
    if show:
        plt.show()
    else:
        plt.close(fig)