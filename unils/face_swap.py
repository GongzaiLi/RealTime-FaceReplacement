import cv2
import numpy as np
from unils.delaunay_triangulation import calculate_delaunay_triangles
from unils.warp_trianglation import affine_transformation
import unils.helper as hf



def face_swap(img_ref, detector, predictor, face_landmark_number):
    # color set
    gray = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale frame
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

    delaunay_triangles = calculate_delaunay_triangles(rect, hull2)

    if len(delaunay_triangles) == 0:
        return None

    affine_transformation(hull1, hull2, delaunay_triangles, img_ref, img_warped)

    # todo refactor here 2 =================================================================================================

    # Calculate Mask
    hull8U = []
    for i in range(0, len(hull2)):
        hull8U.append((hull2[i][0], hull2[i][1]))

    mask = np.zeros(img_ref.shape, dtype=img_ref.dtype)  # https://numpy.org/doc/stable/reference/arrays.dtypes.html

    cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

    # todo what is that
    r = cv2.boundingRect(np.float32([hull2]))

    # todo tuple add tuple
    center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))

    # Clone seamlessly.
    output = cv2.seamlessClone(np.uint8(img_warped), img_ref, mask, center, cv2.NORMAL_CLONE)
    # todo refactor here 2 =================================================================================================

    img_warped = np.copy(img_ref)
    delaunay_triangles = calculate_delaunay_triangles(rect, hull1)

    if len(delaunay_triangles) == 0:
        return None

    # todo refactor here 2 =================================================================================================
    affine_transformation(hull2, hull1, delaunay_triangles, img_ref, img_warped)

    # Calculate Mask
    hull8U = []
    for i in range(0, len(hull2)):
        hull8U.append((hull1[i][0], hull1[i][1]))

    mask = np.zeros(img_ref.shape, dtype=img_ref.dtype)

    cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

    r = cv2.boundingRect(np.float32([hull1]))

    center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))

    # Clone seamlessly.
    output = cv2.seamlessClone(np.uint8(img_warped), output, mask, center, cv2.NORMAL_CLONE)
    # todo refactor here 2 =================================================================================================

    return output
