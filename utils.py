import sys
import cv2
import numpy as np


def get_fourcc(cap: cv2.VideoCapture) -> str:
    """Return the 4-letter string of the codec the video uses.

    Parameters
        cap (cv2.VideoCapture): the OpenCV video capture object

    Returns
        (str): the fourcc codec of the mp4 video
    """
    fourcc_codec = int(cap.get(cv2.CAP_PROP_FOURCC))
    return fourcc_codec.to_bytes(4, byteorder=sys.byteorder).decode()


def get_top_left_coords(corners: list[list]):
    """Finds the top left corner coordinates in a list of rectagle vertices (x, y).

    The origin is the top left corner of the video frame, so this function uses that to determine
    the top left corner of the rectangle. Whichever corner is closest to the origin, is the top left.

    Parameters
        corners (list[list]): list of the rectangle's vertices

    Returns
        top_left_ind (int): index of the top left corner coordinates
    """
    min_dists = [distance_2d(x, y) for x, y in corners]
    return corners[np.argmin(min_dists)]


def distance_2d(x1: float, y1: float, x2: float = 0, y2: float = 0):
    """Calculates the distance between any two points in 2D space.

    Parameters
        x1 (float): x-coordinate of the first point
        y1 (float): y-coordinate of the first point
        x2 (float): x-coordinate of the second point
        y2 (float): y-coordinate of the second point

    Returns
        dist (float): distance between the points
    """
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
