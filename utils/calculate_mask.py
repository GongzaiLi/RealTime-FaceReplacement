import cv2
import numpy as np


def get_calculate_mask(hull, hull_index, img_ref, img_warped):
    """
    calculate face mask
    """
    # Calculate Mask
    hull_points = []
    for i in range(0, len(hull_index)):
        hull_points.append((hull[i][0], hull[i][1]))

    mask = np.zeros(img_ref.shape, dtype=img_ref.dtype)  # https://numpy.org/doc/stable/reference/arrays.dtypes.html

    cv2.fillConvexPoly(mask, np.int32(hull_points), (255, 255, 255))
    r = cv2.boundingRect(np.float32([hull]))
    center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))

    # Clone seamlessly.
    output = cv2.seamlessClone(np.uint8(img_warped), img_ref, mask, center, cv2.NORMAL_CLONE)
    return output
