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


def calculateDelaunayTriangles(rect, points):

    subdiv = cv2.Subdiv2D(rect);

def face_swap3(img_ref, detector, predictor):
    # color set
    gray1 = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale frame
    rects1 = detector(gray1, 0)

    if len(rects1) < 2:
        return None
    print(gray1.shape, 111111111111)
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

    for index in  range(0, len(hullIndex)):
        hull1.append(points1[int(hullIndex[index])])
        hull2.append(points2[int(hullIndex[index])])

    # Find delanauy traingulation for convex hull points
    sizeImg2 = img_ref.shape
    #               height      weight
    rect = (0, 0, sizeImg2[1], sizeImg2[0])

    dt =





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

        # cv2.imshow('frame', frame)  # Show the image on the display.
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  ### add 1

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Close the script when q is pressed.
            break

    # Release the camera device and close the GUI.
    video_capture.release()
    cv2.destroyAllWindows()
