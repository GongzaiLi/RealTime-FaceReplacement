import cv2
import numpy as np


# Apply affine transform calculated using srcTri and dstTri to src and
def apply_affine_transform(src, src_tri, dst_tri, size):
    # Given a pair of triangles, find the affine transform.
    warp_mat = cv2.getAffineTransform(np.float32(src_tri), np.float32(dst_tri))

    # Apply the Affine Transform just found to the src image
    # https://docs.opencv.org/3.4/d4/d61/tutorial_warp_affine.html
    dst = cv2.warpAffine(src, warp_mat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    return dst


def warp_triangle(face1, face2, triangle1, triangle2):
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([triangle1]))
    r2 = cv2.boundingRect(np.float32([triangle2]))

    # Offset points by left top corner of the respective rectangles
    t1_rect = []
    t2_rect = []
    t2_rect_array = []

    for i in range(0, 3):
        t1_rect.append(((triangle1[i][0] - r1[0]), (triangle1[i][1] - r1[1])))
        t2_rect.append(((triangle2[i][0] - r2[0]), (triangle2[i][1] - r2[1])))
        t2_rect_array.append(((triangle2[i][0] - r2[0]), (triangle2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r2[3], r2[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(t2_rect_array), (1.0, 1.0, 1.0), 16, 0)

    # Apply warpImage to small rectangular patches
    img1_rect = face1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]

    size = (r2[2], r2[3])

    img2_rect = apply_affine_transform(img1_rect, t1_rect, t2_rect, size)

    img2_rect = img2_rect * mask

    face2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = face2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] * (
            (1.0, 1.0, 1.0) - mask)

    face2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] = face2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]] + img2_rect


def affine_transformation(hull1, hull2, delaunay_triangles, img_ref, img_warped):
    # Apply affine transformation to Delaunay triangles
    for i in range(0, len(delaunay_triangles)):
        t1 = []
        t2 = []
        # get points for img1, img2 corresponding to the triangles
        for j in range(0, 3):
            t1.append(hull1[delaunay_triangles[i][j]])
            t2.append(hull2[delaunay_triangles[i][j]])

        # warp Triangle
        warp_triangle(img_ref, img_warped, t1, t2)
