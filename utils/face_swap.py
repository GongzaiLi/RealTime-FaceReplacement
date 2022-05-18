import cv2
import numpy as np
from utils.delaunay_triangulation import calculate_delaunay_triangles
from utils.warp_trianglation import affine_transformation
import utils.helper as hf
from utils.calculate_mask import get_calculate_mask


def face_swap(img_ref, detector, predictor, face_landmark_number):

    gray = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 0)

    if len(faces) < 2:
        return None

    if hf.check_is_out_of_image(faces, gray.shape[1], gray.shape[0]):
        return None

    img_warped = np.copy(img_ref)

    # face 1
    face1_shape = hf.get_face_shape(predictor, gray, faces[0], face_landmark_number)
    if face1_shape is None:
        return None
    shape1, points1 = face1_shape

    # face 2
    face2_shape = hf.get_face_shape(predictor, gray, faces[1], face_landmark_number)
    if face2_shape is None:
        return None
    shape2, points2 = face2_shape

    hull1, hull2, rect = hf.get_convex_hull(img_ref, points1, points2)

    # face 2
    delaunay_triangles = calculate_delaunay_triangles(rect, hull2)
    if len(delaunay_triangles) == 0:
        return None

    affine_transformation(hull1, hull2, delaunay_triangles, img_ref, img_warped)
    output = get_calculate_mask(hull2, hull2, img_ref, img_warped)

    img_warped = np.copy(img_ref)

    # face 1
    delaunay_triangles = calculate_delaunay_triangles(rect, hull1)
    if len(delaunay_triangles) == 0:
        return None

    affine_transformation(hull2, hull1, delaunay_triangles, img_ref, img_warped)
    output = get_calculate_mask(hull1, hull2, output, img_warped)

    return output
