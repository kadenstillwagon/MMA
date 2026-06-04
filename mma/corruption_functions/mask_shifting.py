import numpy as np
from copy import deepcopy

def shift_masks(masks, shift_mag_min=2, shift_mag_max=20, shift_prob=1.0, iter=1):
    result = [masks]
    curr_masks = {}
    for label in np.sort(np.unique(masks)):
        if label == 0: continue
        curr_masks[label] = (masks == label)

    for _ in range(iter):
        for mask_label in np.unique(masks):
            if mask_label == 0: continue
            if np.random.random() > shift_prob: continue #does nothing if shift_prob=1.0
            curr_mask = curr_masks[mask_label]

            shift = None
            shift_mag = np.random.randint(shift_mag_min, shift_mag_max)

            right_left = np.random.random() > 0.5
            if right_left:
                left = np.random.random() > 0.5
                if left:
                    shift = (0, -shift_mag)
                else:
                    shift = (0, shift_mag)
            else:
                up = np.random.random() > 0.5
                if up:
                    shift = (-shift_mag, 0)
                else:
                    shift = (shift_mag, 0)

            padded_shifted_mask = np.roll(np.pad(curr_mask.astype(np.uint8), pad_width=shift_mag), shift=shift, axis=(0, 1), ).astype(np.bool)
            shifted_mask = padded_shifted_mask[shift_mag:-shift_mag, shift_mag:-shift_mag]

            curr_masks[mask_label] = shifted_mask

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