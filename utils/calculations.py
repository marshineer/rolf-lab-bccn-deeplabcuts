import sys
import cv2
import numpy as np

MIN_POSITION = 0.1


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


def get_basis_vectors(
        reference_coordinates: dict[int, np.ndarray],
        index: int,
        reference_tag_ids: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Calculates the basis vectors representing the apparatus frame.

    These basis vectors represent a frame of reference defined by the origin and two points.

    Parameters
        reference_coordinates (dict[int, np.ndarray]): top-left corner coordinates of the three reference AprilTags
        index (int): video frame index to calculate for
        reference_tag_ids (list[int]): AprilTag IDs used as reference points (origin, y-dir, x-dir)

    Returns
        basis_v1 (np.ndarray): basis vector in the x-direction
        basis_v2 (np.ndarray): basis vector in the y-direction
    """

    origin, point_y, point_x = reference_tag_ids
    basis_v1 = np.array([reference_coordinates[point_x][0, index] - reference_coordinates[origin][0, index],
                         reference_coordinates[point_x][1, index] - reference_coordinates[origin][1, index]])
    basis_v2 = np.array([reference_coordinates[point_y][0, index] - reference_coordinates[origin][0, index],
                         reference_coordinates[point_y][1, index] - reference_coordinates[origin][1, index]])

    return basis_v1, basis_v2


def get_hand_speed_session(time: np.ndarray, hand_position: np.ndarray) -> list[dict[int, np.ndarray]]:
    """Calculates the x, y and total speed of the transformed hand position data.

    Parameters
        time (np.ndarray): time vectors for each block of a session
        hand_position (np.ndarray): x and y hand landmark position data for all blocks

    Returns
        hand_speed (list[dict[int, np.ndarray]]): x, y and total hand landmark speed for all blocks
    """

    hand_speed = []
    for block_time, block_pos in zip(time, hand_position):
        lm_speed_dict = {}
        for lm, pos in block_pos.items():
            lm_speed_dict[lm] = np.zeros((3, block_time.size - 1))
            for i in range(pos.shape[0]):
                lm_speed_dict[lm][i, :] = calculate_speed(pos[i, :], block_time)
            lm_speed_dict[lm][-1, :] = np.sqrt(lm_speed_dict[lm][0, :] ** 2 + lm_speed_dict[lm][1, :] ** 2)
        hand_speed.append(lm_speed_dict)

    return hand_speed


def calculate_speed(time: np.ndarray, position: np.ndarray) -> np.ndarray:
    """Calculates speed from position data.

    Parameters
        time (np.ndarray): time vector associated with the postion data
        position (np.ndarray): single dimension positon data

    Returns
        (np.ndarray): vector of speed at each time point
    """

    return np.diff(position) / np.diff(time)
