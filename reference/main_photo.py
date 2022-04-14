import cv2
import numpy as np
import dlib


def extract_index_nparray(nparray):
    """
    :param nparray:
    :return:
    """
    index = None
    for num in nparray[0]:
        index = num
        break
    return index


if __name__ == '__main__':
    img1 = cv2.imread("../imgs/test1.jpg")
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

    img2 = cv2.imread("../imgs/test2.jpg")
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    # create a all zero matrix img_gray # create empty mask
    mask = np.zeros_like(img1_gray)  # todo do i need more

    # find faces
    detector = dlib.get_frontal_face_detector()

    # face points
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

    # todo part 5 init 2 new face
    height_1, width_1, channels_1 = img1.shape
    img1_new_face = np.zeros((height_1, width_1, channels_1), np.uint8)

    height_2, width_2, channels_2 = img2.shape
    img2_new_face = np.zeros((height_2, width_2, channels_2), np.uint8)

    faces = detector(img1_gray)
    for face in faces:
        landmarks = predictor(img1_gray, face)
        landmarks_points = []
        # take 68 dots are from detector
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x, y))

            # draw the read dot in the img
            # cv2.circle(img, (x, y), 3, (0, 0, 255), -1)

        # find area and make numpy array
        points = np.array(landmarks_points, np.int32)
        convexhull = cv2.convexHull(points)

        # face make point to the line step 2
        # the area is a mask we need take it
        # cv2.polylines(img, [convexhull], True, (255, 0, 0), 3)

        # step 5 make a white mask
        cv2.fillConvexPoly(mask, convexhull, 255)

        # step 5 mask area puts in image
        face_image_1 = cv2.bitwise_and(img1, img1, mask=mask)

        # =========================================part 2=========================================
        # Delaunay Triangulation
        rect = cv2.boundingRect(convexhull)
        # make a rectangle area
        # (x, y, w, h) = rect
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0))
        subdiv = cv2.Subdiv2D(rect)
        subdiv.insert(landmarks_points)
        # create many triangles
        triangles = subdiv.getTriangleList()

        # make all point in int
        triangles = np.array(triangles, dtype=np.int32)

        index_triangles = []  # part 3

        # t [269. 421. 477. 435. 462. 486.] [p1,x, p1.y, p2.x, p2.y, p3.x, p3.y]


        for t in triangles:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])

            # make line in the photo step_8
            # cv2.line(img, pt1, pt2, (0, 0, 255), 2)
            # cv2.line(img, pt2, pt3, (0, 0, 255), 2)
            # cv2.line(img, pt1, pt3, (0, 0, 255), 2)

            # =============================part 3 ==========================================

            # print(pt1)
            # # setp 9 show the pt1 pt2 and pt3 in the image
            # cv2.circle(img, pt1, 3, (0, 255, 0), -1)
            # cv2.circle(img, pt2, 3, (255, 0, 0), -1)
            # cv2.circle(img, pt3, 3, (0, 0, 255), -1)

            # there is to find where pt1 in all points || some has three number because some pt1 has same x or y
            # then do .all(axis=1)) make one value
            index_pt1 = np.where((points == pt1).all(axis=1))
            index_pt1 = extract_index_nparray(index_pt1)

            index_pt2 = np.where((points == pt2).all(axis=1))
            index_pt2 = extract_index_nparray(index_pt2)

            index_pt3 = np.where((points == pt3).all(axis=1))
            index_pt3 = extract_index_nparray(index_pt3)

            if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                triangle = [index_pt1, index_pt2, index_pt3]
                # add to triangle
                index_triangles.append(triangle)
        print(len(points), len(index_triangles), len(triangles))

    # face 2
    faces2 = detector(img2_gray)

    for face in faces2:
        landmarks = predictor(img2_gray, face)
        landmarks_points2 = []
        # take 68 dots are from detector
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points2.append((x, y))

            # cv2.circle(img2, (x, y), 3, (0, 255, 0), -1)

        points2 = np.array(landmarks_points2, np.int32)
        convexhull2 = cv2.convexHull(points2)

    # Delaunay Triangulation of the second face, from the first face delaunay triangulation
    lines_space_mask = np.zeros_like(img1_gray)
    lines_space_new_face = np.zeros_like(img2)
    print(len(index_triangles), 1111111)
    print(landmarks_points, 222222222222)
    for triangle_index in index_triangles:
        # Triangulation of the first face

        tr1_pt1 = landmarks_points[triangle_index[0]]
        tr1_pt2 = landmarks_points[triangle_index[1]]
        tr1_pt3 = landmarks_points[triangle_index[2]]
        triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

        rect1 = cv2.boundingRect(triangle1)
        (x1, y1, w1, h1) = rect1
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0))
        cropped_triangle = img1[y1: y1 + h1, x1:x1 + w1]

        cropped_tr1_mask = np.zeros((h1, w1), np.uint8)

        points = np.array([[tr1_pt1[0] - x1, tr1_pt1[1] - y1],
                           [tr1_pt2[0] - x1, tr1_pt2[1] - y1],
                           [tr1_pt3[0] - x1, tr1_pt3[1] - y1]], np.int32)

        cv2.fillConvexPoly(cropped_tr1_mask, points, 255)

        # make line in the photo step_8
        # cv2.line(img, tr1_pt1, tr1_pt2, (0, 0, 255), 2)
        # cv2.line(img, tr1_pt2, tr1_pt3, (0, 0, 255), 2)
        # cv2.line(img, tr1_pt1, tr1_pt3, (0, 0, 255), 2)

        lines_space_1 = cv2.bitwise_and(img1, img1, mask=lines_space_mask) # line 137

        # Triangulation of second face
        tr2_pt1 = landmarks_points2[triangle_index[0]]
        tr2_pt2 = landmarks_points2[triangle_index[1]]
        tr2_pt3 = landmarks_points2[triangle_index[2]]

        triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)

        rect2 = cv2.boundingRect(triangle2)
        (x2, y2, w2, h2) = rect2
        # 把所有的小三角做成原图小三角 step 14
        cropped_triangle2 = img1[y2: y2 + h2, x2:x2 + w2]

        cropped_tr2_mask = np.zeros((h2, w2), np.uint8)

        points2 = np.array([[tr2_pt1[0] - x2, tr2_pt1[1] - y2],
                            [tr2_pt2[0] - x2, tr2_pt2[1] - y2],
                            [tr2_pt3[0] - x2, tr2_pt3[1] - y2]], np.int32)

        cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)
        # lines_space_2 = cv2.bitwise_and(img1, img1, mask=lines_space_mask)  # line 137

        # make face 2  creat same triangle with face maybe shape not same but position is same
        # cv2.line(img2, tr2_pt1, tr2_pt2, (0, 0, 255), 2)
        # cv2.line(img2, tr2_pt2, tr2_pt3, (0, 0, 255), 2)
        # cv2.line(img2, tr2_pt1, tr2_pt3, (0, 0, 255), 2)

        # ======================== Warp triangles
        points = np.float32(points)
        points2 = np.float32(points2)

        M = cv2.getAffineTransform(points, points2)
        warped_triangle = cv2.warpAffine(cropped_triangle, M, (w2, h2))  # image 1 to 2
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

        M2 = cv2.getAffineTransform(points2, points)
        warped_triangle2 = cv2.warpAffine(cropped_triangle2, M2, (w1, h1))  # image 2 to 1
        warped_triangle2 = cv2.bitwise_and(warped_triangle2, warped_triangle2, mask=cropped_tr1_mask)
        # break

        # Reconstruct destination face =============================part 5

        # Reconstructing destination face 1
        img1_new_face_rect_area = img1_new_face[y1: y1 + h1, x1:x1 + w1]
        img1_new_face_rect_area_gray = cv2.cvtColor(img1_new_face_rect_area, cv2.COLOR_BGR2GRAY)
        _, mask_triangles_designed_1 = cv2.threshold(img1_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
        warped_triangle2 = cv2.bitwise_and(warped_triangle2, warped_triangle2, mask=mask_triangles_designed_1)

        img1_new_face_rect_area = cv2.add(img1_new_face_rect_area, warped_triangle2)
        img1_new_face[y1: y1 + h1, x1:x1 + w1] = img1_new_face_rect_area



        # Reconstructing destination face 2
        img2_new_face_rect_area = img2_new_face[y2: y2 + h2, x2:x2 + w2]
        img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
        _, mask_triangles_designed_2 = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed_2)

        img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
        img2_new_face[y2: y2 + h2, x2: x2 + w2] = img2_new_face_rect_area

    # Face Swapped (putting 1st face into 2nd face) and (putting 2nd face into 1st face)

    # Face swapped (putting 2nd face into 1st face)
    img1_face_mask = np.zeros_like(img1_gray)
    img1_head_mask = cv2.fillConvexPoly(img1_face_mask, convexhull, 255)
    img1_face_mask = cv2.bitwise_not(img1_head_mask)

    img1_head_noface = cv2.bitwise_and(img1, img1, mask=img1_face_mask)
    result1 = cv2.add(img1_head_noface, img1_new_face)

    (x, y, w, h) = cv2.boundingRect(convexhull)
    center_face1 = (int((x + x + w) / 2), int((y + y + h) / 2))

    seamlessclone_1 = cv2.seamlessClone(result1, img1, img1_head_mask, center_face1, cv2.NORMAL_CLONE)

    # Face swapped (putting 1st face into 2nd face)
    img2_face_mask = np.zeros_like(img2_gray)
    img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
    img2_face_mask = cv2.bitwise_not(img2_head_mask)

    img2_head_noface = cv2.bitwise_and(img2, img2, mask=img2_face_mask)
    result2 = cv2.add(img2_head_noface, img2_new_face)

    (x, y, w, h) = cv2.boundingRect(convexhull2)
    center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))

    seamlessclone_2 = cv2.seamlessClone(result2, img2, img2_head_mask, center_face2, cv2.NORMAL_CLONE)

    cv2.imshow("Image 1", img1)
    cv2.imshow("Image 2", img2)
    # cv2.imshow("Face image 1", face_image_1)
    # cv2.imshow("Mask", mask)
    # cv2.imshow("cropped triangle 1", cropped_triangle)
    # cv2.imshow("cropped triangle 2", cropped_triangle2)
    # cv2.imshow("mask cropped triangle", cropped_tr1_mask)
    # cv2.imshow("Warped triangle", warped_triangle)
    # cv2.imshow("Warped triangle2", warped_triangle2)  # image 2 to 1

    # cv2.imshow("Image 1 new face", img1_new_face)
    # cv2.imshow("Image 2 new face", img2_new_face)
    # cv2.imshow("Background 1", background1)
    # cv2.imshow("Background 2", background2)

    cv2.imshow("seamlessclone 1", seamlessclone_1)
    cv2.imshow("seamlessclone 2", seamlessclone_2)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
