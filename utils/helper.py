import cv2
import numpy as np


def check_is_out_of_image(rects, img_wight, img_height):
    """
    check the faces is in the webcam
    """
    for rect in rects:
        x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
        if x < 0 or y < 0 or (y + h) >= img_height or (x + w) >= img_wight:
            return True
    return False


def check_is_out_of_image_points(points, img_wight, img_height):
    """
    check the faces is in the webcam
    """
    for x, y in points:
        if x < 0 or y < 0 or y >= img_height or x >= img_wight:
            return True
    return False


def get_landmark_points(shape, face_landmark_number):
    """
    get facial landmark points
    """
    landmarks_points = []
    for n in range(0, face_landmark_number):
        x = shape.part(n).x
        y = shape.part(n).y
        landmarks_points.append((x, y))
    return landmarks_points


def get_face_shape(predictor, gray, face, face_landmark_number):
    """
    get face shape
    :return face shape and face points
    """
    shape = predictor(gray, face)

    # points1 = face_utils.shape_to_np(shape1)  # type is an array of arrays
    landmarks_points = get_landmark_points(shape, face_landmark_number)

    if check_is_out_of_image_points(landmarks_points, gray.shape[1],
                                    gray.shape[0]):  # check if points are inside the image
        return None

    # need to covert to a list of tuple
    # map in python3 is return an iterable || map in python2 is return a list
    points = list(map(tuple, landmarks_points))

    return shape, points


def get_convex_hull(img_ref, points1, points2):
    """
    create convex hull in face
    """
    # hull done
    hull1 = []
    hull2 = []

    hull_index = cv2.convexHull(np.array(points2, np.int32), returnPoints=False)
    for i in range(0, len(hull_index)):
        hull1.append(points1[int(hull_index[i])])
        hull2.append(points2[int(hull_index[i])])

    # Find delanauy traingulation for convex hull points
    size_img = img_ref.shape
    rect = (0, 0, size_img[1], size_img[0])

    return hull1, hull2, rect
