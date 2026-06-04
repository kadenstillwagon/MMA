import numpy as np
from copy import deepcopy
import cv2
from skimage.draw import disk

def add_masks(masks, min_radius=6, max_radius=20, num_add_per_iter=10, iter=1):
    rng = np.random.default_rng()

    result = [masks]
    new_masks = deepcopy(masks)
    for _ in range(iter):
        for _ in range(num_add_per_iter):
            radius = rng.integers(min_radius, max_radius, endpoint=True)

            background = (new_masks == 0).astype(np.uint8)
            background[:, 0] = 0
            background[:, -1] = 0
            background[0, :] = 0
            background[-1, :] = 0

            background_dist = cv2.distanceTransform(background, cv2.DIST_L2, maskSize=5)

            valid = background_dist >= radius
            valid_coords_temp = np.column_stack(np.where(valid))
            valid_coords = []
            for coord in valid_coords_temp:
                if coord[0] >= radius and coord[1] >= radius and coord[0] < masks.shape[0] - radius and coord[1] < masks.shape[1] - radius:
                    valid_coords.append(coord)

            if len(valid_coords) == 0:
                if np.max(background_dist) >= min_radius or min_radius == max_radius:
                    radius = int(np.max(background_dist))

                    valid = background_dist >= radius
                    valid_coords_temp = np.column_stack(np.where(valid))
                    valid_coords = []
                    for coord in valid_coords_temp:
                        if coord[0] >= radius and coord[1] >= radius and coord[0] < masks.shape[0] - radius and coord[1] < masks.shape[1] - radius:
                            valid_coords.append(coord)
                    
                    if len(valid_coords) == 0:
                        continue
                else:
                    continue
            
            y, x = valid_coords[np.random.randint(len(valid_coords))]

            rr, cc = disk((y, x), radius)
            new_masks[rr, cc] = np.max(np.unique(new_masks)) + 1
            
        result.append(deepcopy(new_masks))
    
    return result