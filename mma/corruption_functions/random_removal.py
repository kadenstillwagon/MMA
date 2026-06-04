import numpy as np
from copy import deepcopy


def remove_masks(masks, remove_prob=0.2, iter=1):
    result = [masks]
    new_masks = deepcopy(masks)
    for _ in range(iter):

        removals_to_make = int(len(np.unique(new_masks)) * remove_prob)

        for _ in range(removals_to_make):
            label_to_remove = np.random.choice(np.unique(new_masks), 1)

            mask = (new_masks == label_to_remove)
            
            new_masks[mask] = 0
            
        result.append(deepcopy(new_masks))
    
    return result