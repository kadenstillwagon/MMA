import numpy as np

#########################
#   SEG IMPLEMENTATION
#########################
from mma.metrics.matching import create_overlap_dict


def get_accuracy_plot(pred, gt, matched_dict):
    accuracy_plot = np.where(gt > 0, 1, 0)
    accuracy_plot[pred > 0] = 1

    for gt_label, pred_labels in matched_dict.items():
        for pred_label in pred_labels:
            accuracy_plot[(pred == pred_label) & (gt == gt_label)] = 2

    return accuracy_plot

def compute_jaccard_scores(pred, gt, overlap_dict, threshold=0.5):

    matched_dict = {}

    jaccard_scores = []
    for gt_label, pred_overlaps in overlap_dict.items():
        gt_idx = int(gt_label[3:])
        R = np.sum(np.where(gt == gt_idx, 1, 0))

        max_jaccard = 0
        max_jaccard_idx = None
        for pred_label, overlap_area in pred_overlaps.items(): 
            pred_idx = int(pred_label[4:])

            RnS = overlap_area
            S = np.sum(np.where(pred == pred_idx, 1, 0))
            RuS = R + S - RnS
            jaccard = RnS / (RuS + 1e-5)
            # print(RnS, threshold * R)
            if RnS > threshold * R:
                if jaccard > max_jaccard:
                    max_jaccard = jaccard
                    max_jaccard_idx = pred_idx

        jaccard_scores.append(max_jaccard)
        if max_jaccard_idx is not None:
            try:
                matched_dict[gt_idx].append(max_jaccard_idx)
            except:
                matched_dict[gt_idx] = [max_jaccard_idx]

    mean_jaccard = np.mean(jaccard_scores)

    return mean_jaccard, matched_dict


def get_seg(pred, gt, threshold=0.5, return_accuracy_plot=False):
    overlap_dict = create_overlap_dict(pred, gt)

    seg, matched_dict = compute_jaccard_scores(pred, gt, overlap_dict, threshold)

    if return_accuracy_plot:
        accuracy_plot = get_accuracy_plot(pred, gt, matched_dict)
        
        return seg, accuracy_plot
    else:
        return seg





