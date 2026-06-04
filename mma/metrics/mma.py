import numpy as np
from mma.metrics.matching import get_mask_associations


#ACCURACY PLOT
def get_accuracy_plot(pred, gt, associated_masks):
    accuracy_plot = np.where(gt > 0, 1, 0)
    accuracy_plot[pred > 0] = 1

    for association in associated_masks:
      if 'GT' in association[0]:
        gt_label = int(association[0][3:])
        pred_label = int(association[1][4:])
      else:
        gt_label = int(association[1][3:])
        pred_label = int(association[0][4:])

      accuracy_plot[(pred == pred_label) & (gt == gt_label)] = 2

    return accuracy_plot


#DENOMINATOR
def calculate_denominator(pred, gt):
  """
  Calculates the denominator of accuracy metric as the union between the area
  covered by the ground truth segmenetations and the area covered by model
  segmentations
  Author(s): Kaden Stillwagon

  args:
      pred (np.ndarray): Array of shape (W,H) representing model segmentation.
      gt (np.ndarray): Array of shape (W,H) representing the ground truth segmentation.

  Returns:
      denominator (int): Integer representing the number of pixels coverd by the
                          union between the GT and model segmentations

  """

  #Get segmentation indices
  gt_indices = np.argwhere(gt > 0)
  pred_indices = np.argwhere(pred > 0)

  #Convert to sets
  gt_indices_set = set(tuple(row) for row in gt_indices)
  pred_indices_set = set(tuple(row) for row in pred_indices)

  #Calculate Union
  union_gt_pred_indices = gt_indices_set | pred_indices_set

  return len(union_gt_pred_indices)



#NUMERATOR
def calculate_total_matched_mask_overlap(associated_masks, overlap_dict):
  total_area = 0

  for association in associated_masks:
    if 'GT' in association[0]:
      gt_label = association[0]
      pred_label = association[1]
    else:
      gt_label = association[1]
      pred_label = association[0]

    overlap_area = overlap_dict[gt_label][pred_label]

    total_area += overlap_area

  return total_area


def calculate_numerator(pred, gt, greedy):

  #Associate GT and Output Segmentations (Max Matching)
  associated_masks, overlap_dict = get_mask_associations(
    pred=pred,
    gt=gt,
    greedy=greedy
  )

  #Calculate Total Overlap Area of GT and Output Segmentation
  numerator = calculate_total_matched_mask_overlap(
      associated_masks=associated_masks,
      overlap_dict=overlap_dict
  )

  return numerator, associated_masks


#ACCURACY

def compute_accuracy_max_matching(pred, gt, greedy):

  #Calculate Denominator (Union of GT and output segmentations)
  denominator = calculate_denominator(
      pred=pred,
      gt=gt
  )

  #Calculate Numerator (Total Area of GT and Associated Segmentation Overlap)
  numerator, associated_masks = calculate_numerator(
      pred=pred,
      gt=gt,
      greedy=greedy
  )

  mma = numerator / (denominator + 1e-5)

  return mma, associated_masks



def get_mma(pred, gt, greedy=False, return_accuracy_plot=False):

  #Compute mma
  mma, associated_masks = compute_accuracy_max_matching(pred, gt, greedy)

  if return_accuracy_plot:
    accuracy_plot = get_accuracy_plot(pred, gt, associated_masks)

    return mma, accuracy_plot
  else:
    return mma

