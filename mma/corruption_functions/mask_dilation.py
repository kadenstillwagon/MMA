import numpy as np
from copy import deepcopy
import cv2

def dilate_masks(masks, iter=1):
    result = [masks]
    curr_masks = {}
    for label in np.sort(np.unique(masks)):
        if label == 0: continue
        curr_masks[label] = (masks == label)

    for _ in range(iter):
        for mask_label in np.unique(masks):
            if mask_label == 0: continue
            
            curr_mask = curr_masks[mask_label]

            padded_dilated_mask = cv2.dilate(np.pad(curr_mask.astype(np.uint8), pad_width=1), kernel=np.ones((3, 3)), iterations=1).astype(np.bool)
            dilated_mask = padded_dilated_mask[1:-1, 1:-1]

            curr_masks[mask_label] = dilated_mask

        new_masks = np.zeros_like(masks)
        labels = []
        counts = []
        for label in curr_masks.keys():
            labels.append(label)
            counts.append(np.sum(curr_masks[label].astype(np.uint8)))
        
        for idx in reversed(np.argsort(counts)):
            label = labels[idx]
            if label == 0: continue
            new_masks[curr_masks[label]] = label    
            
        result.append(deepcopy(new_masks))
    
    return result