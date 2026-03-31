""" Given a set of 3D points, render them from a given camera pose and intrinsics. """


import numpy as np
import cv2
import os
from pathlib import Path


def render_points(img, points, intrinsics, distort, rot, trans, img_size=(1920, 1080)):
    """ Render the given 3D points from the given camera pose and intrinsics. """
    rvec = cv2.Rodrigues(rot)[0]
    # Project the 3D points to 2D using the intrinsics and pose

    projected_points, _ = cv2.projectPoints(points, rvec, trans, intrinsics, distort)
    projected_points = projected_points.reshape(-1, 2)

    # Draw the projected points on the image
    for point in projected_points:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < img_size[0] and 0 <= y < img_size[1]:
            cv2.circle(img, (x, y), 8, (0, 0, 255), -1)

    # Optionally, draw lines between the points to visualize the structure
    # For example, if the points form a cube, we can connect them accordingly
    connections = [(0, 1), (1, 2), (2, 3), (3, 0),  # Base square
                   (4, 5), (5, 6), (6, 7), (7, 4),  # Top square
                   (0, 4), (1, 5), (2, 6), (3, 7)]  # Vertical lines

    for start, end in connections:
        pt1 = (int(projected_points[start][0]), int(projected_points[start][1]))
        pt2 = (int(projected_points[end][0]), int(projected_points[end][1]))
        if all(0 <= pt[0] < img_size[0] and 0 <= pt[1] < img_size[1] for pt in [pt1, pt2]):
            cv2.line(img, pt1, pt2, (0, 255, 255), 2)

    return img


if __name__ == "__main__":

    ids = ['104122061649', '950122060411']
    # ids = ['950122061707']
    dir_path = os.path.dirname(os.path.realpath(__file__))

    for id in ids:
        images_path = dir_path + f"/data/pose/images/{id}"
        calib_file = dir_path + f"/output/calib/{id}/calib.yaml"

        file = cv2.FileStorage(calib_file, cv2.FILE_STORAGE_READ)
        intrinsics = file.getNode("intrinsic").mat()
        distortion = file.getNode("distortion").mat()
        rot = file.getNode("Rot").mat()
        trans = file.getNode("Trans").mat()

        # 8 points in square in mm
        scale = 60
        points = np.array([[0, 0, 0],
                           [1, 0, 0],
                           [1, 1, 0],
                           [0, 1, 0],
                           [0, 0, -1],
                           [1, 0, -1],
                           [1, 1, -1],
                           [0, 1, -1]], dtype=np.float32)
        points = points * scale

        all_images = Path(images_path).glob("*.png")
        cv2.namedWindow("Rendered Points", cv2.WINDOW_KEEPRATIO)
        for image in all_images:
            img = cv2.imread(image)
            img = render_points(img, points, intrinsics, distortion, rot, trans, img_size=(img.shape[1], img.shape[0]))
            cv2.imshow("Rendered Points", img)
            cv2.waitKey(0)
            break
