import cv2
import numpy as np
from copy import deepcopy


def get_border_coords(mask):
    """
    Get Border Coords
    Finds the coordinates of all of the border pixels
    Author(s): Kaden Stillwagon

    Args:
      mask (np.ndarray): Array of shape (W,H) representing a binary mask of the cell. 1 = cell, 0 = background.


    Returns:
      coords (list): List of xy coordinates for each border pixel 
    """
    mask = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    coords = []
    for contour in contours:
        coords.extend([(int(pt[0][1]), int(pt[0][0])) for pt in contour])  # (y, x) format
    
    coords = np.array(coords)
    row_coords, col_coords = coords[:, 0], coords[:, 1]
    return row_coords, col_coords



################
#   EROSION
################
def erode_masks(masks, iter=1):
    results = [deepcopy(masks)]
    for i in range(iter):
        mask_indices = np.unique(masks)

        eroded_masks = np.zeros_like(masks)
        for idx in mask_indices:
            mask_img = np.where(masks == idx, 1, 0)
            border_row_coords, border_col_coords = get_border_coords(mask_img)
            mask_img[border_row_coords, border_col_coords] = 0
            eroded_masks += mask_img * idx

        masks = eroded_masks
        results.append(deepcopy(masks))

    return results
