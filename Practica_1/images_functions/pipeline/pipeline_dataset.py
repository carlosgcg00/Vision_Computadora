# pipeline.py
import os
import json
import cv2
from tqdm import tqdm

# --- Project imports (adjust paths to your repo structure) ---
from images_functions.data_io.read_imgs_directory import list_images_in_directory
from images_functions.data_io.read_data import read_image_from_cv2
from images_functions.data_io.save_data import save_image_cv2
from images_functions.utils.extract_channels import decompose_image_channels

from images_functions.pipeline.preprocess_pipeline import pipeline_preprocess_img
from images_functions.image_binarization.contour_analysis import find_contours_img_binarization
from images_functions.evaluate_segmentations.evaluation_masks import masks_iou, visualize_iou, cm_binary_imgs, visualize_segmentation_confusion
# -------------------------------------------------------------


def run_pipeline(
    img_path: str,
    img_path_label: str,
    steps: list,
    step_name: str,
    area_max = False,
    area_relative_min: float = 0.01,
    plot_flag: bool = False,
    channel_img:str = 'img_gray' 
):
    """
    Runs the complete segmentation pipeline:
      - Iterates through all images in img_path
      - Saves original RGB image
      - Applies preprocessing pipeline
      - Extracts predicted contours
      - Extracts GT contours from label masks
      - Computes IoU (GT vs Pred)
      - Visualizes IoU overlays on original image
      - Saves metrics into output/<step_name>/results.json

    Args:
        img_path (str): Path to color images.
        img_path_label (str): Path to ground truth masks.
        steps (list): Processing pipeline [(func, "title", {params}), ...].
        step_name (str): Output subfolder name.
        area_relative_min (float): Minimum relative area for predicted contours.
        plot_flag (bool): Whether to show intermediate plots interactively.
    """
    out_dir = os.path.join("output", step_name)
    os.makedirs(out_dir, exist_ok=True)
    results_path = os.path.join(out_dir, "results.json")

    # Load existing results
    if os.path.exists(results_path):
        try:
            with open(results_path, "r") as f:
                results_data = json.load(f)
        except json.JSONDecodeError:
            results_data = []
    else:
        results_data = []

    processed_images = {r.get("image_name") for r in results_data}

    list_imgs = list_images_in_directory(img_path)
    print(f"The directory {img_path} contains {len(list_imgs)} images.")

    for img_selected in tqdm(list_imgs, desc=f"Running {step_name}", unit="image"):
        img_base_name = os.path.basename(img_selected)

        if img_base_name in processed_images:
            continue

        try:
            # --- Load and decompose image ---
            image = decompose_image_channels(img_selected)
            img_channel = image[channel_img]
            img_rgb = image['img_rgb']

            # --- Define output paths ---
            output_figure = os.path.join(out_dir, os.path.splitext(img_base_name)[0])
            os.makedirs(output_figure, exist_ok=True)

            save_img_name = os.path.join(output_figure, img_base_name)
            save_plot_img_preprocessed = os.path.join(output_figure, "preprocessed.jpg")
            save_plot_img_contours = os.path.join(output_figure, "contours.jpg")
            save_plot_img_iou = f'{output_figure}/IoU_img_mask.jpg'
            save_plot_contour_iou = f'{output_figure}/IoU_contour_mask.jpg'
            # Save original image
            save_image_cv2(img_rgb, save_img_name)

            # --- Preprocess image ---
            result_img, _ = pipeline_preprocess_img(
                img=img_channel,
                steps=steps,
                plot_flag=plot_flag,
                save_plot=save_plot_img_preprocessed
            )

            # --- Predicted contours ---
            _, _, _, _, contour_masks_pred, result_img = find_contours_img_binarization(
                image=img_channel,
                binarized_image=result_img,
                area_max=area_max,
                area_relative_min=area_relative_min,
                plot_flag=plot_flag,
                save_plot=save_plot_img_contours
            )

            # --- Ground truth contours ---
            img_selected_label = os.path.join(img_path_label, img_base_name)
            image_label = read_image_from_cv2(img_selected_label, mode=cv2.IMREAD_GRAYSCALE)

            _, _, _, _, contour_masks_gt, _ = find_contours_img_binarization(
                image=image_label,
                binarized_image=image_label,
                area_relative_min=0.0,
                plot_flag=False,
                save_plot=None
            )

            # --- IoU computation image ---
            results = cm_binary_imgs(image_label, result_img)
            visualize_segmentation_confusion(image['img_rgb'], image_label, result_img, save_plot=save_plot_img_iou, plot_flag=False)

            # --- IoU computation per contour mask ---
            matches, iou_values, iou_matrix, mean_iou = masks_iou(
                masks_gt=contour_masks_gt,
                masks_pred=contour_masks_pred
            )

            # --- IoU visualization ---
            visualize_iou(
                image=img_rgb,
                masks_gt=contour_masks_gt,
                masks_pred=contour_masks_pred,
                matches=matches,
                save_plot=save_plot_contour_iou,
                plot_flag=plot_flag
            )

            # --- Append to results.json ---
            new_entry = {
                "image_name": img_base_name,
                "acc_image": results['acc'],
                "iou_image": results['iou'],
                "iou_values_contour_mask": iou_values,
            }
            results_data.append(new_entry)
            with open(results_path, "w") as f:
                json.dump(results_data, f, indent=4)

        except Exception as e:
            print(f"Error processing {img_base_name}: {e}")


