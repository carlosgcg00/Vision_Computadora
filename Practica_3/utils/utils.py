from PIL import Image
import os
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def save_ds_images(dataset, labels, save_dir):
    """
    Save the dataset images in a given directory.
    
    Args:
        dataset: Dataset to save.
        labels: List of labels.
        save_dir: Directory to save the images.
    """
    
    
    os.makedirs(save_dir, exist_ok=True)
    counter = {label: 0 for label in labels}
    
    for sample in dataset:
        image = sample["image"]
        file_name = f"{labels[sample['labels']]}_{counter[sample['labels']]:04d}.png"
        file_path = os.path.join(save_dir, file_name)
        
        try:
            if isinstance(image, Image.Image):
                image.save(file_path)
            elif isinstance(image, np.ndarray):
                Image.fromarray(image).save(file_path)
            counter[sample['label']] += 1
        except Exception as e:
            print(f"Error saving image {file_path}: {e}")

def add_ds_image_size(example, img_label = "image"):
    """
    Add the image size to the dataset.
    
    Args:
        example: Example to add the image size.
        img_label: Label of the image.
    """
    
    img = example[img_label]
    if isinstance(img, Image.Image):
        example["width"] = img.width
        example["height"] = img.height
        # example["resolution"] = img.width * img.height
    elif isinstance(img, np.ndarray):
        example["width"] = img.shape[1]
        example["height"] = img.shape[0]
        # example["resolution"] = img.shape[1] * img.shape[0]

    return example


def show_N_images(dataset, labels, N=5, N_cols=2, save_dir = None, save_name = None, sup_title = None):
    """
    Show N images from the dataset.
    
    Args:
        dataset: Dataset to show the images.
        labels: List of labels.
        N: Number of images to show.
        N_cols: Number of columns.
        save_dir: Directory to save the images.
        save_name: Name of the saved image.
        sup_title: Super title.
    """
    
    N_rows = math.ceil(N / N_cols)

    fig, axes = plt.subplots(
        N_rows,
        N_cols,
        figsize=(5 * N_cols, 5 * N_rows)
    )

    # Asegurar indexación uniforme
    axes = axes.flatten() if N > 1 else [axes]

    for i in range(N):
        image = dataset[i]["image"]
        label = dataset[i]["labels"]

        axes[i].imshow(image)
        axes[i].set_title(f"{labels[label]} | Size: {image.size}")
        axes[i].axis("off")

    # Desactivar subplots sobrantes
    for j in range(N, len(axes)):
        axes[j].axis("off")

    if sup_title:
        plt.suptitle(sup_title)

    plt.tight_layout()
    plt.show()
    if save_dir and save_name:
        plt.savefig(os.path.join(save_dir, save_name), dpi=300)

def show_one_example_per_label(dataset, N_cols=4):
    """
    Show exactly one example per label.
    
    Args:
        dataset: Hugging Face Dataset
        N_cols: Number of columns in the grid
    """
    labels_feature = dataset.features["labels"]
    label_names = labels_feature.names
    num_labels = len(label_names)

    N_rows = math.ceil(num_labels / N_cols)

    fig, axes = plt.subplots(
        N_rows,
        N_cols,
        figsize=(4 * N_cols, 4 * N_rows)
    )

    axes = axes.flatten()

    for label_id, label_name in enumerate(label_names):
        # Buscar el primer índice con esa etiqueta
        idx = next(
            i for i, y in enumerate(dataset["labels"]) if y == label_id
        )

        image = dataset[idx]["image"]

        axes[label_id].imshow(image)
        axes[label_id].set_title(
            f"{label_name}\nSize: {image.size}",
            fontsize=10
        )
        axes[label_id].axis("off")

    # Apagar subplots sobrantes
    for j in range(num_labels, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()



def analysis_dataset(dataset, labels):
    """
    Analyze the dataset.
    
    Args:
        dataset: Dataset to analyze.
        labels: List of labels.
    """
    
    class_counter = {label: 0 for label in labels}
    for sample in dataset:
        label_id = sample["labels"]
        label_name = labels[label_id]

        class_counter[label_name] += 1

    df = pd.DataFrame(class_counter.items(), columns=["Label", "Count"]).sort_values(by="Count", ascending=False)
    return df



def plot_loss_accuracy(history, save_dir = None, save_name = None):
    """
    Plot the loss and accuracy of the model.
    
    Args:
        history: History of the model.
        save_dir: Directory to save the plot.
        save_name: Name of the saved plot.
    """
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    axs[0].plot(range(0, len(history.history['accuracy'])), history.history['accuracy'], label='Train')
    axs[0].plot(range(0, len(history.history['val_accuracy'])), history.history['val_accuracy'], label='Test')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()

    axs[1].plot(range(0, len(history.history['loss'])), history.history['loss'], label='Train')
    axs[1].plot(range(0, len(history.history['val_loss'])), history.history['val_loss'], label='Test')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Loss')
    axs[1].legend()
    
    
    plt.show()
    if save_dir and save_name:
        plt.savefig(os.path.join(save_dir, save_name), dpi=300)

def plot_loss_accuracy_tl_ft(history_tf, history_ft, save_dir = None, save_name = None):
    """
    Plot the loss and accuracy of the model.
    
    Args:
        history: History of the model.
        save_dir: Directory to save the plot.
        save_name: Name of the saved plot.
    """
    
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    axs[0].plot(range(0, len(history_tf.history['accuracy'])), history_tf.history['accuracy'], label='Train (TF)')
    axs[0].plot(range(0, len(history_tf.history['val_accuracy'])), history_tf.history['val_accuracy'], label='Test (TF)')
    axs[0].plot(range(0, len(history_ft.history['accuracy'])), history_ft.history['accuracy'], label='Train (FT)')
    axs[0].plot(range(0, len(history_ft.history['val_accuracy'])), history_ft.history['val_accuracy'], label='Test (FT)')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()

    axs[1].plot(range(0, len(history_tf.history['loss'])), history_tf.history['loss'], label='Train (TF)')
    axs[1].plot(range(0, len(history_tf.history['val_loss'])), history_tf.history['val_loss'], label='Test (TF)')
    axs[1].plot(range(0, len(history_ft.history['loss'])), history_ft.history['loss'], label='Train (FT)')
    axs[1].plot(range(0, len(history_ft.history['val_loss'])), history_ft.history['val_loss'], label='Test (FT)')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Loss')
    axs[1].legend()
    
    
    plt.show()
    if save_dir and save_name:
        plt.savefig(os.path.join(save_dir, save_name), dpi=300)