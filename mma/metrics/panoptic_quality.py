import numpy as np


def get_accuracy_plot(pred, gt, matched_dict):
    accuracy_plot = np.where(gt > 0, 1, 0)
    accuracy_plot[pred > 0] = 1

    for gt_label, pred_label in matched_dict.items():
        accuracy_plot[(pred == pred_label) & (gt == gt_label)] = 2

    return accuracy_plot


def get_true_positives_50(pred, gt):
  TP = 0
  matched_dict = {}

  pred_labels = np.unique(pred)

  for label in pred_labels:
    if label != 0:
        mask_coords = np.argwhere(pred == label)
        gt_at_mask_coords = gt[mask_coords[:, 0], mask_coords[:, 1]]
        gt_intersection_labels, gt_intersection_label_counts = np.unique(gt_at_mask_coords, return_counts=True)
        for i in range(len(gt_intersection_labels)):
            pred_mask_size = len(mask_coords)
            gt_mask_size = len(np.argwhere(gt == gt_intersection_labels[i]))

            intersection = gt_intersection_label_counts[i]
            union = pred_mask_size + gt_mask_size - intersection
            iou = intersection / union

            if iou >= 0.5:
                TP += 1
                matched_dict[gt_intersection_labels[i]] = label
                break

  return TP, matched_dict


def get_segmentation_quality_score(pred, gt, matched_dict):
    TP = len(matched_dict.keys())
    sum_matched_IoU = 0.0

    for gt_label, pred_label in matched_dict.items():
        pred_mask = np.where(pred == pred_label, 1, 0)
        gt_mask = np.where(gt == gt_label, 1, 0)
        intersection = np.sum(pred_mask * gt_mask)
        union = np.sum(pred_mask) + np.sum(gt_mask) - intersection

        iou = intersection / union
        sum_matched_IoU += iou

    sq = sum_matched_IoU / TP
    
    return sq

def get_pq(pred, gt, return_accuracy_plot=False):
    try:
        num_pred_masks = len(np.unique(pred)) - 1
        TP, matched_dict = get_true_positives_50(pred, gt)
        if len(matched_dict.keys()) > 0:
            FP = num_pred_masks - TP
            FN = len(np.unique(gt)) - 1 - TP

            rq = TP / (TP + (0.5 * FP) + (0.5 * FN))
            sq = get_segmentation_quality_score(pred, gt, matched_dict)

            pq = sq * rq
        else:
            pq = 0.0

        if return_accuracy_plot:
            accuracy_plot = get_accuracy_plot(pred, gt, matched_dict)
            return pq, accuracy_plot
        else:
            return pq
    except Exception as e:
        print(e)
        return 0.0