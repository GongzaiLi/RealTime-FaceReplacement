#! /usr/bin/env python

import cv2
import dlib
import numpy as np
import imutils
from imutils import face_utils


def is_out_of_image(rects, img_wight, img_height):
    for rect in rects:
        x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
        if x < 0 or y < 0 or (y + h) >= img_wight or (x + w) >= img_height:
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


def calculateDelaunayTriangles(rect, points):
    # create subdiv
    subdiv = cv2.Subdiv2D(rect)

    # todo need check Insert points into subdiv
    for point in points:
        subdiv.insert(point)

    # todo check the function is right
    triangleList = subdiv.getTriangleList()

    delaunayTri = []

    count = 0

    # todo I do not understand why did that
    for t in triangleList:
        pt = []
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        # todo update
        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            count += 1
            ind = []  # todo check it
            for j in range(0, 3):
                for k in range(0, len(points)):
                    if abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0:
                        ind.append(k)
            if len(ind) == 3:
                delaunayTri.append((ind[0], ind[1], ind[2]))
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

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    # img2Rect = np.zeros((r2[3], r2[2]), dtype = img1Rect.dtype)

    size = (r2[2], r2[3])

    img2Rect = applyAffineTransform(img1, t1Rect, t2Rect, size)

    img2Rect = img2Rect * mask

    # todo  Copy triangular region of the rectangular patch to the output image ???????????????????????? because is numpy
    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (
            (1.0, 1.0, 1.0) - mask)

    img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2Rect


def face_swap3(img_ref, detector, predictor):
    # color set
    gray1 = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale frame
    rects1 = detector(gray1, 0)

    if len(rects1) < 2:
        return None
    if is_out_of_image(rects1, gray1.shape[1], gray1.shape[0]):
        return None

    img1Warped = np.copy(img_ref)

    # todo face 1

    shape1 = predictor(gray1, rects1[0])

    # todo check the face_utils shape_to_np
    points1 = face_utils.shape_to_np(shape1)  # type is an array of arrays

    if is_out_of_image_points(points1, gray1.shape[1], gray1.shape[0]):
        return None

    # need to covert to a list of tuple
    # map in python3 is return an iterable || map in python2 is return a list
    points1 = list(map(tuple, points1))

    # todo face 2
    shape2 = predictor(gray1, rects1[1])
    points2 = face_utils.shape_to_np(shape2)

    if is_out_of_image_points(points2, gray1.shape[1], gray1.shape[0]):  # check if points are inside the image
        return None

    points2 = list(map(tuple, points2))

    # todo Find convex hull
    hull1 = []
    hull2 = []

    hullIndex = cv2.convexHull(np.array(points2), returnPoints=False)

    for index in range(0, len(hullIndex)):
        hull1.append(points1[int(hullIndex[index])])
        hull2.append(points2[int(hullIndex[index])])

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

        # get points for img1, img2 corresponding to the triangles
        for j in range(0, 3):
            t1.append(hull1[i][j])
            t2.append(hull2[i][j])

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
            t1.append(hull1[i][j])
            t2.append(hull2[i][j])

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


if __name__ == '__main__':
    # version check
    # a = cv2.__version__
    # print(a)

    detector = dlib.get_frontal_face_detector()

    # Take face mode
    model = "models/shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(model)

    # cam
    video_path = -1  # if 0 not work go -1
    video_capture = cv2.VideoCapture(video_path)  # Open the first camera connected to the computer.

    # ret, img = video_capture.read()
    # cv2.imshow("Face Swapped", img)
    # img = cv2.imread('ji2.jpg')
    # img2 = cv2.imread('donald_trump.jpg')

    while True:
        ret, img = video_capture.read()  # Read an image from the frame.

        output = face_swap3(img, detector, predictor)
        if (output != None):
            cv2.imshow("Face Swapped", output)
        else:
            cv2.imshow("Face Swapped", img)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Close the script when q is pressed.
            break

    # Release the camera device and close the GUI.
    video_capture.release()
    cv2.destroyAllWindows()
