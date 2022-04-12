#! /usr/bin/env python

import cv2
import dlib

def is_out_of_image(rects, img_wight, img_height):
    for rect in rects:
        x, y, w, h = rect.left(), rect.top(), rect.width(), rect.height()
        if x < 0 or y <0 or (y+h) >= img_wight or (x+w) >= img_height:
            return True
    return False

def face_swap3(img_ref, detector, predictor):

    # color set
    gray1 = cv2.cvtColor(img_ref, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale frame
    rects1 = detector(gray1, 0)

    if len(rects1) < 2:
        return None

    if is_out_of_image(rects1, gray1.shape[1], gray1.shape[0]):
        return None




if __name__ == '__main__':
    # version check
    # a = cv2.__version__
    # print(a)

    detector = dlib.get_frontal_face_detector()

    # Take face mode
    model = "models/shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(model)

    # cam
    video_path = 0  # if 0 not work go -1
    video_capture = cv2.VideoCapture(video_path)  # Open the first camera connected to the computer.

    # ret, img = video_capture.read()
    # cv2.imshow("Face Swapped", img)
    # img = cv2.imread('ji2.jpg')
    # img2 = cv2.imread('donald_trump.jpg')

    while True:
        ret, img = video_capture.read()

        output = face_swap3(img, detector, predictor)
