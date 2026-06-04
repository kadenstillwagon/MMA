import numpy as np
from mma.metrics.matching import create_overlap_dict


def get_accuracy_plot(pred, gt, matched_dict):
    accuracy_plot = np.where(gt > 0, 1, 0)
    accuracy_plot[pred > 0] = 1

    for gt_label, pred_labels in matched_dict.items():
        for pred_label in pred_labels:
            accuracy_plot[(pred == pred_label) & (gt == gt_label)] = 2

    return accuracy_plot


def compute_aji(pred, gt):
    C = 0
    U = 0

    matched_dict = {}

    used_pred_labels = []

    overlap_dict = create_overlap_dict(pred, gt)

    for gt_label, pred_overlaps in overlap_dict.items():
        i = int(gt_label[3:])
        G_i = np.sum(np.where(gt == i, 1, 0))

        max_jaccard = 0
        max_jaccard_label = None
        max_jaccard_intersection = None
        max_jaccard_union = None
        for pred_label, overlap_area in pred_overlaps.items(): 
            k = int(pred_label[4:])

            G_i_n_S_k = overlap_area
            S_k = np.sum(np.where(pred == k, 1, 0))
            G_i_u_S_k = G_i + S_k - G_i_n_S_k
            jaccard = G_i_n_S_k / (G_i_u_S_k + 1e-5)
            
            if jaccard > max_jaccard:
                max_jaccard = jaccard
                max_jaccard_label = k
                max_jaccard_intersection = G_i_n_S_k
                max_jaccard_union = G_i_u_S_k

        if max_jaccard_label is not None:
            C = C + max_jaccard_intersection
            U = U + max_jaccard_union
            used_pred_labels.append(max_jaccard_label)
            try:
                matched_dict[i].append(max_jaccard_label)
            except:
                matched_dict[i] = [max_jaccard_label]
        else:
            U = U + G_i

    #Account for unmatched model predictions
    for k in np.unique(pred):
        if k != 0:
            if k not in used_pred_labels:
                S_k = np.sum(np.where(pred == k, 1, 0))

                U = U + S_k

                used_pred_labels.append(k)

    A = C / U

    return A, matched_dict


def get_aji(pred, gt, return_accuracy_plot=False):
    aji, matched_dict = compute_aji(pred, gt)

    if return_accuracy_plot:
        accuracy_plot = get_accuracy_plot(pred, gt, matched_dict)
        
        return aji, accuracy_plot
    else:
        return aji
            