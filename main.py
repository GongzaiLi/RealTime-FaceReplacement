#! /usr/bin/env python
import cv2
import dlib
from utils.face_swap import face_swap


def pick_face_landmark_number():
    while True:
        input_model = input("Input 1 to choose 68 face landmarks module \n"
                            "Input 2 to choose 81 face landmarks module \n"
                            "Then press ENTER button\n")
        if input_model == '1' or input_model == '2':
            face_landmark_number = 68 if input_model == '1' else 81
            return face_landmark_number
        print('Input error.')


if __name__ == '__main__':

    face_landmark_number = pick_face_landmark_number()

    # Take face mode
    model = f"models/shape_predictor_{face_landmark_number}_face_landmarks.dat"

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(model)

    video_path = -1  # if 0 not work go -1
    video_capture = cv2.VideoCapture(video_path)  # Open the first camera connected to the computer.

    print("Pressing q to Stop")

    while True:
        ret, img = video_capture.read()  # Read an image from the frame.
        output = face_swap(img, detector, predictor, face_landmark_number)

        if output is not None:
            cv2.imshow("Face Replacement output", output)
        else:
            cv2.imshow("Face Replacement input", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Close the script when q is pressed.
            break

    cv2.destroyAllWindows()
    video_capture.release()
