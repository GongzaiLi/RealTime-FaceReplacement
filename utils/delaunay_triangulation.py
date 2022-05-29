import cv2
import numpy as np


def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index


def calculate_delaunay_triangles(rect, points):
    """
    create delaunay triangles in a face
    """
    # create sub div
    sub_div = cv2.Subdiv2D(rect)
    sub_div.insert(points)

    triangle_list = sub_div.getTriangleList()
    triangle_list = np.array(triangle_list, dtype=np.int32)
    np_points = np.array(points, np.int32)

    delaunay_triangles = []

    for t in triangle_list:

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
            delaunay_triangles.append(triangle)

    return delaunay_triangles
