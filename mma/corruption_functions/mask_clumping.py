import numpy as np
from copy import deepcopy
from mma.parameter_calculations import find_center_v5, crop_to_mask
import matplotlib.pyplot as plt

# def clump_masks(masks, clump_prob=0.2, iter=1):
#     result = [masks]
#     new_masks = deepcopy(masks)
#     for _ in range(iter):

#         merges_to_make = int(len(np.unique(new_masks)) * clump_prob)

#         for _ in range(merges_to_make):
#             label_to_merge = np.random.choice(np.unique(new_masks), 1)

#             mask = (new_masks == label_to_merge)
#             cropped_mask, _, crop_coords = crop_to_mask(mask, mask)

#             mask_center = find_center_v5(cropped_mask)
#             center_y = mask_center[0] + crop_coords[0]
#             center_x = mask_center[1] + crop_coords[2]

#             closest_label = None
#             min_min_dist = np.inf
#             min_avg_dist = np.inf
#             for mask_idx in np.unique(new_masks):
#                 if mask_idx == 0: continue
#                 if mask_idx == label_to_merge: continue
#                 y_coords, x_coords = np.where(new_masks == mask_idx)
#                 dists = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
#                 min_dist = np.min(dists)
#                 avg_dist = np.mean(dists)

#                 if min_dist < min_min_dist:
#                     closest_label = mask_idx
#                     min_min_dist = min_dist
#                     min_avg_dist = avg_dist
#                 elif min_dist == min_min_dist:
#                     if avg_dist < min_avg_dist:
#                         closest_label = mask_idx
#                         min_avg_dist = avg_dist

#             merge_mask = (new_masks == closest_label)

#             new_masks[merge_mask] = label_to_merge
            
#         result.append(deepcopy(new_masks))
    
#     return result


def clump_masks(masks, clump_prob=0.2, iter=1):
    result = [masks.copy()]
    new_masks = masks.copy()

    for _ in range(iter):

        labels = np.unique(new_masks)
        labels = labels[labels != 0]

        merges_to_make = int(len(labels) * clump_prob)

        # --- Precompute centroids once ---
        centroids = {}

        for label in labels:
            ys, xs = np.where(new_masks == label)
            centroids[label] = np.array([
                ys.mean(),
                xs.mean()
            ])

        labels_array = np.array(list(centroids.keys()))
        centroid_array = np.stack([centroids[l] for l in labels_array])

        for _ in range(merges_to_make):

            label_to_merge = np.random.choice(labels_array)

            center = centroids[label_to_merge]

            # Compute centroid distances to all labels at once
            dists = np.linalg.norm(
                centroid_array - center,
                axis=1
            )

            # Exclude self
            self_idx = np.where(labels_array == label_to_merge)[0][0]
            dists[self_idx] = np.inf

            closest_label = labels_array[np.argmin(dists)]

            # Merge
            new_masks[new_masks == closest_label] = label_to_merge

            # Update centroid approximately
            ys, xs = np.where(new_masks == label_to_merge)
            new_center = np.array([ys.mean(), xs.mean()])

            centroids[label_to_merge] = new_center

            # Remove merged label
            del centroids[closest_label]

            keep = labels_array != closest_label
            labels_array = labels_array[keep]
            centroid_array = centroid_array[keep]

            merge_idx = np.where(labels_array == label_to_merge)[0][0]
            centroid_array[merge_idx] = new_center

        result.append(new_masks.copy())

    return result