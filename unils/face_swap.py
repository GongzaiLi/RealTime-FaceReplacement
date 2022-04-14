import cv2
import numpy as np
from unils.delaunay_triangulation import calculate_delaunay_triangles
from unils.warp_trianglation import warp_triangle


def is_out_of_image(rects, img_wight, img_height):
    for rect in rects:
        x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
        if x < 0 or y < 0 or (y + h) >= img_height or (x + w) >= img_wight:
            return True
    return False


def is_out_of_image_points(points, img_wight, img_height):
    for x, y in points:
        if x < 0 or y < 0 or y >= img_height or x >= img_wight:
            return True
    return False


def get_landmark_points(shape, face_landmark_number):
    landmarks_points = []
    for n in range(0, face_landmark_number):
        x = shape.part(n).x
        y = shape.part(n).y
        landmarks_points.append((x, y))
    return landmarks_points


def face_swap(img_ref, detector, predictor, face_landmark_number):
    # color set
    gray1 = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale frame
    faces = detector(gray1, 0)

    if len(faces) < 2:
        return None

    if is_out_of_image(faces, gray1.shape[1], gray1.shape[0]):
        return None

    img1_warped = np.copy(img_ref)

    # todo refactor here 1 =================================================================================================
    # face 1
    shape1 = predictor(gray1, faces[0])

    # points1 = face_utils.shape_to_np(shape1)  # type is an array of arrays
    landmarks_1_points = get_landmark_points(shape1, face_landmark_number)

    if is_out_of_image_points(landmarks_1_points, gray1.shape[1], gray1.shape[0]):
        return None

    # need to covert to a list of tuple
    # map in python3 is return an iterable || map in python2 is return a list
    points1 = list(map(tuple, landmarks_1_points))


    # face 2
    shape2 = predictor(gray1, faces[1])
    # points2 = face_utils.shape_to_np(shape2)
    landmarks_2_points = get_landmark_points(shape2, face_landmark_number)

    if is_out_of_image_points(landmarks_2_points, gray1.shape[1],
                              gray1.shape[0]):  # check if points are inside the image
        return None

    points2 = list(map(tuple, landmarks_2_points))
    # todo refactor here 1 =================================================================================================

    # hull done
    hull1 = []
    hull2 = []

    hullIndex = cv2.convexHull(np.array(points2, np.int32), returnPoints=False)
    for i in range(0, len(hullIndex)):
        hull1.append(points1[int(hullIndex[i])])
        hull2.append(points2[int(hullIndex[i])])

    # Find delanauy traingulation for convex hull points
    sizeImg2 = img_ref.shape
    #               height      weight
    rect = (0, 0, sizeImg2[1], sizeImg2[0])

    delaunayTri = calculate_delaunay_triangles(rect, hull2)

    if len(delaunayTri) == 0:
        return None


    # todo refactor here 2 =================================================================================================
    for i in range(0, len(delaunayTri)):
        t1 = []
        t2 = []

        # get points for img1, img2 corresponding to the triangles todo------------------------------------
        for j in range(0, 3):
            t1.append(hull1[delaunayTri[i][j]])
            t2.append(hull2[delaunayTri[i][j]])

        # warp Triangle
        warp_triangle(img_ref, img1_warped, t1, t2)



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
    output = cv2.seamlessClone(np.uint8(img1_warped), img_ref, mask, center, cv2.NORMAL_CLONE)
    # todo refactor here 2 =================================================================================================


    img1_warped = np.copy(img_ref)
    delaunayTri = calculate_delaunay_triangles(rect, hull1)



    if len(delaunayTri) == 0:
        return None


    # todo refactor here 2 =================================================================================================
    for i in range(0, len(delaunayTri)):
        t1 = []
        t2 = []

        # get points for img1, img2 corresponding to the triangles
        for j in range(0, 3):
            t1.append(hull2[delaunayTri[i][j]])
            t2.append(hull1[delaunayTri[i][j]])

        # warp Triangle
        warp_triangle(img_ref, img1_warped, t1, t2)




    # Calculate Mask
    hull8U = []
    for i in range(0, len(hull2)):
        hull8U.append((hull1[i][0], hull1[i][1]))

    mask = np.zeros(img_ref.shape, dtype=img_ref.dtype)

    cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

    r = cv2.boundingRect(np.float32([hull1]))

    center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))

    # Clone seamlessly.
    output = cv2.seamlessClone(np.uint8(img1_warped), output, mask, center, cv2.NORMAL_CLONE)
    # todo refactor here 2 =================================================================================================

    return output
