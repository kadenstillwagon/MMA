import numpy as np
import networkx as nx
import copy


def create_overlap_dict(pred, gt):
  overlap_dict = {}

  gt_labels = np.unique(gt)[1:]

  for label in gt_labels:
    overlap_dict[f'GT_{int(label)}'] = {}
    mask_locations = np.argwhere(gt == label)
    pred_labels_at_mask = pred[mask_locations[:, 0], mask_locations[:, 1]]
    pred_labels, pred_label_counts = np.unique(pred_labels_at_mask, return_counts=True)
    for i in range(len(pred_labels)):
      if pred_labels[i] != 0:
        overlap_dict[f'GT_{int(label)}'][f'OUT_{int(pred_labels[i])}'] = int(pred_label_counts[i])

  return overlap_dict


def get_max_overlap_matching(overlap_dict):
  #Turn GT, model output segmentation matching into maximum weight bipartite matching problem
  #One set of nodes GT for ground truth segs, one set of nodes OUT for output segs
  #Edges between GT and OUT nodes are weight by their area of overlap
  #Max weight bipartite matching will return a matching between GT and OUT such
  #that each OUT segmentation can be assigned to at most 1 GT segmentation and
  #resulting matching will be the matching with highest overlap area

  #Create networkx graph
  G = nx.Graph()

  #Create edges between GT segs and every output seg that overlaps with overlap area as edge weight
  for gt_label, pred_overlaps in overlap_dict.items():
    for pred_label, overlap_area in pred_overlaps.items():
      G.add_edge(gt_label, pred_label, weight=overlap_area)

  #Compute the maximum weight matching of G
    #Matching - subset of edges where no node occurs more that once
    #Weight of matching sum of the weights of its edges
    #Maximal matching cannot add more edges and still be a matching
  max_overlap_matching = nx.max_weight_matching(G, maxcardinality=False)

  return max_overlap_matching



def get_greedy_overlap_matching(overlap_dict):
  greedy_overlap_matching = set()

  used_pred_labels = []

  for gt_label, pred_overlaps in overlap_dict.items():
    max_overlap = 0
    max_pred_label = None
    for pred_label, overlap_area in pred_overlaps.items():
      if pred_label not in used_pred_labels:
        if overlap_area > max_overlap:
          max_overlap = overlap_area
          max_pred_label = pred_label

    if max_pred_label is not None:
      greedy_overlap_matching.add((max_pred_label, gt_label))
      used_pred_labels.append(max_pred_label)

  return greedy_overlap_matching




def get_mask_associations(pred, gt, greedy=False):

  #Calculate Overlap Dictionary
  overlap_dict = create_overlap_dict(
      pred=pred,
      gt=gt
  )

  #Associate GT and Output Segmentations (Max Matching)
  if not greedy:
    associated_masks = get_max_overlap_matching(
      overlap_dict=copy.deepcopy(overlap_dict)
    )
  else:
    associated_masks = get_greedy_overlap_matching(
      overlap_dict=copy.deepcopy(overlap_dict)
    )

  return associated_masks, overlap_dict