import numpy as np

def fragment_masks(masks, frag_prob=0.3, iter=1):
    result = [masks]
    for _ in range(iter):
        new_masks = np.zeros_like(masks)

        mask_labels, label_counts = np.unique(masks, return_counts=True)
        viable_labels = mask_labels[label_counts >= 25]

        mask_indices_to_fragment = np.random.choice(a=viable_labels, size=int((len(viable_labels) - 1) * frag_prob), replace=False)
        
        next_new_index = np.max(np.unique(masks)) + 1

        for mask_label in np.unique(masks):
            if mask_label == 0: continue

            mask = (masks == mask_label)

            if mask_label in mask_indices_to_fragment:
                y_coords, x_coords = np.where(mask)
                peak_coord_indices = np.random.choice(len(y_coords), 2)
                peak_1 = (y_coords[peak_coord_indices[0]], x_coords[peak_coord_indices[0]])
                peak_2 = (y_coords[peak_coord_indices[1]], x_coords[peak_coord_indices[1]])

                dist_1 = np.sqrt((y_coords - peak_1[0])**2 + (x_coords - peak_1[1])**2)
                dist_2 = np.sqrt((y_coords - peak_2[0])**2 + (x_coords - peak_2[1])**2)

                new_fragments = np.zeros_like(masks)

                indices_closer_to_peak_1 = dist_1 <= dist_2
                new_fragments[y_coords[indices_closer_to_peak_1], x_coords[indices_closer_to_peak_1]] = 1
                new_fragments[y_coords[~indices_closer_to_peak_1], x_coords[~indices_closer_to_peak_1]] = 2

                for frag_label in np.unique(new_fragments):
                    if frag_label == 0: continue
                    fragment_mask = (new_fragments == frag_label)
                    if frag_label == 1:
                        new_masks[fragment_mask] = mask_label
                    else:
                        new_masks[fragment_mask] = next_new_index
                        next_new_index += 1
            else:
                new_masks[mask] = mask_label
            
            
        masks = new_masks
        result.append(new_masks)
    
    return result