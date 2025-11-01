import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

  



# === Shared palette (same RGBs in both visualizers, in [0,1]) ===
COLOR_TP = np.array([0.00, 1.00, 0.00])   # green
COLOR_TN = np.array([0.53, 0.81, 0.98])   # light blue (≈ sky blue)
COLOR_FN = np.array([1.00, 0.00, 0.00])   # red
COLOR_FP = np.array([0.00, 0.00, 0.55])   # dark blue (≈ navy)

def to_boolean(maskA, maskB):
    """ Transform both masks into boolean values. (If they are img_binarized they supposed to be booleans).
    Furthermore we are goin to assure both masks have the same shape, otherwise they cannot be compared"""
    if maskA.shape != maskB.shape:
        raise ValueError(f"Both masks must have same shape. \nMask A: {maskA.shape} != MaskB: {maskB.shape}")
    return maskA.astype(bool), maskB.astype(bool)

def get_tp_tn_fp_fn(maskA, maskB):
    maskA, maskB = to_boolean(maskA , maskB)

    tp_mask = np.logical_and(maskA,  maskB)
    tn_mask = np.logical_and(~maskA, ~maskB)
    fp_mask = np.logical_and(~maskA,  maskB)
    fn_mask = np.logical_and(maskA,  ~maskB)

    return tp_mask, tn_mask, fp_mask, fn_mask

def cm_binary_imgs(maskA, maskB):
    """ Compute TP, TN, FP, FN 
    Args:
        -   maskA: Boolean mask
        -   maskB: Boolean mask
    Return:
        - Dict with the TP, TN, FP, FN
    """

    tp, tn, fp, fn = get_tp_tn_fp_fn(maskA, maskB)
    
    tp = tp.sum()
    tn = tn.sum()
    fp = fp.sum()
    fn = fn.sum()

    total = tp+tn+fn+fp
    acc = (tp+tn)/total if total > 0 else 0.0
    iou = tp / (tp+fp+fn) if (tp+fp+fn) > 0 else 0.0

    confusion_matrix = np.array([[tp, fn], [fp, tn]], dtype = int) 

    return {'TP': int(tp), 'TN': int(tn), 'FP': int(fp), 'FN': int(fn), 'acc': float(acc), 'iou': float(iou), 'confusion_matrix': confusion_matrix}


def visualize_segmentation_confusion(img_original, img_bin_label, img_bin, plot_flag=True, save_plot=None, alpha=0.4):
    """
    Visualize a confusion overlay for binary segmentation on top of the original image.

    Colors:
      - TP: green
      - TN: light blue
      - FN: red
      - FP: dark blue

    The function does not return anything; it only displays and/or saves the figure.

    Args:
        img_original (H,W,3): Base image (uint8 or float in [0,1]).
        img_bin_label (H,W): Ground-truth binary mask (bool or {0,1}).
        img_bin (H,W): Predicted binary mask (bool or {0,1}).
        plot_flag (bool): If True, display the figure with plt.show().
        save_plot (str or None): If not None, path to save the figure (e.g., "overlay.png").
        alpha (float): Opacity for the colored overlay (0=transparent, 1=opaque).
    """

    img_show = img_original.astype(np.float32)
    if img_show.max() > 1.0:
        img_show = img_show / 255.0

    tp_mask, tn_mask, fp_mask, fn_mask = get_tp_tn_fp_fn(img_bin_label, img_bin)

    result = cm_binary_imgs(img_bin_label, img_bin)
    tp = result['TP']
    tn = result['TN']
    fp = result['FP']
    fn = result['FN']
    acc = result['acc']
    iou = result['iou']

    # Colors (shared palette)
    color_tp = COLOR_TP      # verde
    color_tn = COLOR_TN      # azul claro
    color_fn = COLOR_FN      # rojo
    color_fp = COLOR_FP      # azul oscuro

    H, W = img_bin_label.shape
    overlay = np.zeros((H, W, 4), dtype=np.float32)  # RGBA

    def paint(mask, rgb, a):
        overlay[mask, :3] = rgb
        overlay[mask,  3] = a

    paint(tp_mask, color_tp, alpha)
    paint(tn_mask, color_tn, alpha)
    paint(fn_mask, color_fn, alpha)
    paint(fp_mask, color_fp, alpha)

    # Plot
    fig = plt.figure(figsize=(8, 8))
    plt.imshow(img_show)
    plt.imshow(overlay)            # overlay con alfa por píxel
    plt.axis('off')
    plt.title(f"Acc={acc:.3f} | IoU={iou:.3f}")

    # Legend
    legend_elems = [
        Patch(facecolor=color_tp, alpha=alpha, label=f"TP"),
        Patch(facecolor=color_tn, alpha=alpha, label=f"TN"),
        Patch(facecolor=color_fn, alpha=alpha, label=f"FN"),
        Patch(facecolor=color_fp, alpha=alpha, label=f"FP"),
    ]
    plt.legend(handles=legend_elems, loc='lower right', frameon=True)

    if save_plot is not None:
        plt.savefig(save_plot, bbox_inches='tight', dpi=200)

    if plot_flag:
        plt.show()
    else:
        plt.close(fig)

    
def compute_iou(maskA, maskB):
    """ Compute intersections/union between two boolean masks o two binarized images. 
    Args:
        - maskA: Boolean mask
        - maskB: Boolean mask
    Returns:
        intersection/union
    """
    intersection = np.logical_and(maskA, maskB).sum()
    union = np.logical_or(maskA, maskB).sum()
    if union == 0:
        return 0.0
    return intersection/union


def compute_iou_array_masks(masks_gt, masks_pred):
    """Compute IoU between all pairs of ground truth and predicted masks
    Args:
        - masks_gt: Array of masks labels
        - masks_pred: Array of masks predicts
    Return: 
        array[leng(gt) x len(pred)] with IoU scores, here we compute the IoU for each gt with each pred
    """
    iou_matrix = np.zeros((len(masks_gt), len(masks_pred)))
    for i, gt in enumerate(masks_gt):
        for j, pred in enumerate(masks_pred):
            iou_matrix[i, j] = compute_iou(gt, pred)
    return iou_matrix    

def masks_iou(masks_gt, masks_pred, iou_threshold=0.0):
    """
    Compute the IoU and Accuracy between each ground truth and predicted mask.
    Matching is done with Hungarian algorithm to maximize IoU (and avoid duplicates).

    Args:
        - masks_gt: list/array of boolean ground-truth masks
        - masks_pred: list/array of boolean predicted masks
        - iou_threshold (float): minimum IoU to consider a valid match

    Returns:
        - matches: [(gt_idx, pred_idx, iou_value, acc_value), ...]
        - iou_values: list of IoU scores for valid matches
        - acc_values: list of Accuracy scores for valid matches
        - iou_matrix: IoU matrix [len(gt), len(pred)]
        - acc_matrix: Accuracy matrix [len(gt), len(pred)]
        - mean_iou: average IoU across valid matches
        - mean_acc: average Accuracy across valid matches
    """
    # --- Matrices de IoU y ACC ---
    n_gt, n_pred = len(masks_gt), len(masks_pred)
    iou_matrix = np.zeros((n_gt, n_pred), dtype=float)

    for i, gt in enumerate(masks_gt):
        for j, pred in enumerate(masks_pred):
            gt_b, pred_b = to_boolean(gt, pred)
            iou_matrix[i, j] = compute_iou(gt_b, pred_b)

    # --- Hungarian Matching (max IoU) ---
    from scipy.optimize import linear_sum_assignment
    cost = 1.0 - iou_matrix
    if iou_threshold > 0.0:
        cost[iou_matrix < iou_threshold] = 1e6

    gt_idxes, pred_idxes = linear_sum_assignment(cost)

    # --- Emparejamiento final ---
    matches, iou_values = [], []
    for gi, pj in zip(gt_idxes, pred_idxes):
        iou_val = iou_matrix[gi, pj]
        if iou_val >= iou_threshold and cost[gi, pj] < 1e6:
            matches.append((gi, pj, float(iou_val)))
            iou_values.append(float(iou_val))

    mean_iou = np.mean(iou_values) if len(iou_values) > 0 else 0.0

    return matches, iou_values, iou_matrix, mean_iou



def visualize_iou(image, masks_gt, masks_pred, matches,
                  alpha_union=0.35, alpha_intersection=0.6,
                  save_plot=None, plot_flag=False):
    """
    Visualize GT–Pred overlaps with mean IoU and Accuracy on the image.

    Colors:
      - Intersection (green): overlapping area (TP)
      - Union exclusive (light blue): area of either GT or Pred only (≈ TN)
      - Background: original image or light gray

    Args:
        - image: optional RGB or grayscale image for context.
        - masks_gt: list of boolean GT masks.
        - masks_pred: list of boolean predicted masks.
        - matches: list of (gt_idx, pred_idx, iou_value, acc_value)
        - alpha_union: opacity for union areas.
        - alpha_intersection: opacity for intersection areas.
        - save_plot: optional path to save.
        - plot_flag: if True, displays figure.
    """
    # Determine base dimensions
    if len(masks_gt) > 0:
        H, W = masks_gt[0].shape
    elif len(masks_pred) > 0:
        H, W = masks_pred[0].shape
    else:
        raise ValueError("No masks provided.")

    # Prepare background
    if image is not None:
        bg = np.asarray(image).astype(np.float32)
        if bg.ndim == 2:
            bg = np.stack([bg, bg, bg], axis=-1)
        if bg.max() > 1.0:
            bg /= 255.0
    else:
        bg = np.ones((H, W, 3), dtype=np.float32) * (220.0 / 255.0)

    # Global unions/intersections
    union_total = np.zeros((H, W), dtype=bool)
    inter_total = np.zeros((H, W), dtype=bool)

    iou_vals= []
    for gt_idx, pred_idx, iou_val in matches:
        gt = masks_gt[gt_idx].astype(bool)
        pred = masks_pred[pred_idx].astype(bool)
        inter = np.logical_and(gt, pred)
        union = np.logical_or(gt, pred)
        inter_total |= inter
        union_total |= union
        iou_vals.append(iou_val)

    union_exclusive = np.logical_and(union_total, ~inter_total)

    overlay = np.zeros((H, W, 4), dtype=np.float32)
    overlay[union_exclusive, :3] = COLOR_TN
    overlay[union_exclusive, 3] = alpha_union
    overlay[inter_total, :3] = COLOR_TP
    overlay[inter_total, 3] = alpha_intersection

    mean_iou = np.mean(iou_vals) if len(iou_vals) > 0 else 0.0

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(bg)
    ax.imshow(overlay)
    ax.axis("off")
    ax.set_title(f"Eval masks of contours \n Mean IoU={mean_iou:.3f} ")

    legend_elems = [
        Patch(facecolor=COLOR_TP, alpha=alpha_intersection, label="Intersection"),
        Patch(facecolor=COLOR_TN, alpha=alpha_union, label="Union"),
    ]
    ax.legend(handles=legend_elems, loc="lower right", frameon=True)

    plt.tight_layout()
    if save_plot:
        plt.savefig(save_plot, bbox_inches="tight", dpi=200)

    if plot_flag:
        plt.show()
    else:
        plt.close(fig)
