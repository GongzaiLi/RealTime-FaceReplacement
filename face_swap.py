#! /usr/bin/env python
import cv2

import numpy as np


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


# Apply affine transform calculated using srcTri and dstTri to src and
# todo output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size):
    # Given a pair of triangles, find the affine transform.
    waroMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # Apply the Affine Transform just found to the src image https://docs.opencv.org/3.4/d4/d61/tutorial_warp_affine.html
    dst = cv2.warpAffine(src, waroMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    return dst


def rectContains(rect, point):
    # Check if a point is inside a rectangle
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[0] + rect[2]:
        return False
    elif point[1] > rect[1] + rect[3]:
        return False
    return True


def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index


def calculateDelaunayTriangles(rect, points):
    # create subdiv
    sub_div = cv2.Subdiv2D(rect)

    # todo need check Insert points into subdiv
    # for point in points:
    #     sub_div.insert(point)
    sub_div.insert(points)

    # todo check the function is right
    triangleList = sub_div.getTriangleList()
    triangleList = np.array(triangleList, dtype=np.int32)
    np_points = np.array(points, np.int32)

    delaunayTri = []

    # todo I do not understand why did that
    for t in triangleList:

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        index_pt1 = np.where((np_points == pt1).all(axis=1))
        index_pt1 = extract_index_nparray(index_pt1)

        index_pt2 = np.where((np_points == pt2).all(axis=1))
        index_pt2 = extract_index_nparray(index_pt2)

        index_pt3 = np.where((np_points == pt3).all(axis=1))
        index_pt3 = extract_index_nparray(index_pt3)

        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            delaunayTri.append(triangle)
    return delaunayTri


# Warps and alpha blends triangular regions from img1 and img2 to img
def warpTriangle(img1, img2, t1, t2):
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    t2RectInt = []

    # todo why here is 3
    for i in range(0, 3):
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))
        t2RectInt.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2RectInt), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    # img2Rect = np.zeros((r2[3], r2[2]), dtype = img1Rect.dtype)

    size = (r2[2], r2[3])

    img2Rect = applyAffineTransform(img1Rect, t1Rect, t2Rect, size)

    img2Rect = img2Rect * mask

    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (
            (1.0, 1.0, 1.0) - mask)

    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2Rect


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

    img1Warped = np.copy(img_ref)  # todo ===================

    # face 1
    shape1 = predictor(gray1, faces[0])

    # points1 = face_utils.shape_to_np(shape1)  # type is an array of arrays
    landmarks_1_points = get_landmark_points(shape1, face_landmark_number)

    if is_out_of_image_points(landmarks_1_points, gray1.shape[1], gray1.shape[0]):
        return None

    # need to covert to a list of tuple
    # map in python3 is return an iterable || map in python2 is return a list
    points1 = list(map(tuple, landmarks_1_points))
    # points1 = np.array(landmarks_1_points, np.int32)

    # face 2
    shape2 = predictor(gray1, faces[1])
    # points2 = face_utils.shape_to_np(shape2)
    landmarks_2_points = get_landmark_points(shape2, face_landmark_number)

    if is_out_of_image_points(landmarks_2_points, gray1.shape[1],
                              gray1.shape[0]):  # check if points are inside the image
        return None

    points2 = list(map(tuple, landmarks_2_points))

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

    # todo new staff

    delaunayTri = calculateDelaunayTriangles(rect, hull2)

    if len(delaunayTri) == 0:
        return None

    # todo Apply affine transformation to Delaunay triangles

    for i in range(0, len(delaunayTri)):
        t1 = []
        t2 = []

        # get points for img1, img2 corresponding to the triangles todo------------------------------------
        for j in range(0, 3):
            t1.append(hull1[delaunayTri[i][j]])
            t2.append(hull2[delaunayTri[i][j]])

        # warp Triangle
        warpTriangle(img_ref, img1Warped, t1, t2)

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
    output = cv2.seamlessClone(np.uint8(img1Warped), img_ref, mask, center, cv2.NORMAL_CLONE)

    # todo =================================================== refactor
    img1Warped = np.copy(img_ref)
    delaunayTri = calculateDelaunayTriangles(rect, hull1)

    if len(delaunayTri) == 0:
        return None

    # todo Apply affine transformation to Delaunay triangles
    for i in range(0, len(delaunayTri)):
        t1 = []
        t2 = []

        # get points for img1, img2 corresponding to the triangles
        for j in range(0, 3):
            t1.append(hull2[delaunayTri[i][j]])
            t2.append(hull1[delaunayTri[i][j]])

        # warp Triangle
        warpTriangle(img_ref, img1Warped, t1, t2)

    # Calculate Mask
    hull8U = []
    for i in range(0, len(hull2)):
        hull8U.append((hull1[i][0], hull1[i][1]))

    mask = np.zeros(img_ref.shape, dtype=img_ref.dtype)

    cv2.fillConvexPoly(mask, np.int32(hull8U), (255, 255, 255))

    r = cv2.boundingRect(np.float32([hull1]))

    center = ((r[0] + int(r[2] / 2), r[1] + int(r[3] / 2)))

    # Clone seamlessly.
    output = cv2.seamlessClone(np.uint8(img1Warped), output, mask, center, cv2.NORMAL_CLONE)

    return output