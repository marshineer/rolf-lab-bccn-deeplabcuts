import os
import sys
import cv2
import numpy as np
import pandas as pd


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


def get_files_containing(start_dir: str, string_match: str, string_exclude: str = "XXXXXXXXXX"):
    """Returns the files containing a particular string and their relative paths from the project root.

    File paths are relative from a given starting directory.

    References:
    - https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory
    - https://stackoverflow.com/questions/6618515/sorting-list-according-to-corresponding-values-from-a-parallel-list

    Parameters
        start_dir (str): directory at which to start the walk
        string_match (str): only get files that contain this string
        string_exclude (str): only get files that do not contain this string

    Returns
        paths (list[str]): sorted list of file directory paths
        files (list[str]): sorted list of the file names
    """
    files = []
    paths = []
    for (dirpath, dirnames, filenames) in os.walk(start_dir):
        for file in filenames:
            if string_match in file and string_exclude not in file:
                files.append(file)
                paths.append(dirpath)
    paths_sorted = [path for _, path in sorted(zip(files, paths))]
    files.sort()
    return paths_sorted, files


def load_diode_data(participant_id: str, session_id: str) -> pd.DataFrame:
    """Loads the preprocessed light diode sensor data as a dataframe.

    Parameters
        participant_id (str): unique participant identifier "PXX"
        session_id (str): session identifier ["A1", "A2", "B1", "B2"]

    Returns
        diode_df (pd.DataFrame): processed diode data
    """
    diode_path = f"data/pipeline_data/{participant_id}/{session_id}/{participant_id:02}_{session_id}_diode_sensor.csv"
    return pd.read_csv(diode_path)


def load_video_time(participant_id: str, session_id: str) -> np.ndarray:
    """Loads the preprocessed light diode sensor data as a dataframe.

    Parameters
        participant_id (str): unique participant identifier "PXX"
        session_id (str): session identifier ["A1", "A2", "B1", "B2"]

    Returns
        diode_df (pd.DataFrame): processed diode data
    """
    time_path = f"data/pipeline_data/{participant_id}/{session_id}/{participant_id:02}_{session_id}_video_time.csv"
    return pd.read_csv(time_path).to_numpy('float')
